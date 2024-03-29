function integrate_fuu!(result::DiffResult, y::Vector{T}, cv_u::CellVectorValues, material::MaterialModels.AbstractMaterial, state::Vector{<:MaterialModels.AbstractMaterialState}, ae::Vector{Float64}) where {T, dim}
    f!(y, ae) = fu!(y, cv_u, material, state, ae)
    result = ForwardDiff.jacobian!(result, f!, y, ae);
    return result
end

function fu!(f::Vector{T}, cv_u::CellVectorValues{dim}, material::MaterialModels.AbstractMaterial, state::Vector{<:MaterialModels.AbstractMaterialState}, ae::Vector{T}) where {dim,T}

    for iqp in 1:getnquadpoints(cv_u)
        
        dV = getdetJdV(cv_u, iqp)
        
        ∇u = function_gradient(cv_u, iqp, ae)
        ε = symmetric(∇u)
        σ, C, state[iqp] = material_response(material, ε, state[iqp])

        for i in 1:getnbasefunctions(cv_u)

            ∇Ni = shape_gradient(cv_u, iqp, i)
            f[i] += (σ ⊡ ∇Ni) * dV
        end 
    end

end

function integrate_fuu2!(ke::Matrix{Float64}, f::Vector{Float64}, cv_u::CellVectorValues{dim}, material::MaterialModels.AbstractMaterial, state::Vector{<:MaterialModels.AbstractMaterialState}, ae::Vector{T}) where {dim,T}
    for iqp in 1:getnquadpoints(cv_u)
        
        dV = getdetJdV(cv_u, iqp)
        
        ∇u = function_gradient(cv_u, iqp, ae)
        ε = symmetric(∇u)
        
        if dim == 2
            σ, C, state[iqp] = material_response(PlaneStrain(), material, ε, state[iqp])
        else
            σ, C, state[iqp] = material_response(material, ε, state[iqp])
        end
        
        for i in 1:getnbasefunctions(cv_u)
            ∇Ni = shape_gradient(cv_u, iqp, i)
            f[i] += (σ ⊡ ∇Ni) * dV
            for j in 1:getnbasefunctions(cv_u)
                ∇Nj = shape_gradient(cv_u, iqp, j)
                ke[i,j] += (∇Nj ⊡ C ⊡ ∇Ni) * dV
            end
        end 
    end

end


function integrate_fλu!(result::DiffResult, y::Vector{T}, cv_u::CellVectorValues, ae::Vector{Float64}, dir::Int) where T
    f!(y, ae) = fλu!(y, cv_u, ae, dir)
    result = ForwardDiff.jacobian!(result, f!, y, ae);
    return result
end

function fλu!(f::Vector{T}, cv_u::CellVectorValues{dim}, ae::Vector{T}, dir::Int) where {dim,T}
    e = basevec(Vec{dim}, dir)
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        u = function_value(cv_u, iqp, ae)
        f[1] += (u ⋅ e) * dV 
    end
end

function integrate_fλu_2!(ke::Matrix{T}, f::Vector{T}, cv_u::CellVectorValues{dim}, ae::Vector{T}, dir::Int) where {dim,T}
    e = basevec(Vec{dim}, dir)
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        u = function_value(cv_u, iqp, ae)
        f[1] += (u ⋅ e) * dV 
        for i in 1:getnbasefunctions(cv_u)
            Ni = shape_value(cv_u, iqp, i)
            ke[1,i] += (Ni ⋅ e) * dV
        end
    end
end


function test_fλu()
    ip_u = Lagrange{2,RefCube,1}()
    qr   = QuadratureRule{2,RefCube}(2)

    cv_u = CellVectorValues(qr, ip_u)

    X = Ferrite.reference_coordinates(ip_u) .*0.5
    reinit!(cv_u, X)

    fλ = zeros(Float64, 1)
    ae = zeros(Float64, getnbasefunctions(cv_u))
    diffresult_λ = DiffResults.JacobianResult(fλ, ae)

    fill!(fλ, 0.0)
    diffresult_λ = integrate_fλθ!(diffresult_λ, fλ, cv_u, X, ae, 1); 
    Ke = DiffResults.jacobian(diffresult_λ)

end

function integrate_fλθ!(result::DiffResult, y::Vector, cv_u::CellVectorValues, X::Vector{Vec{dim,Float64}}, ae::Vector{Float64}, dir::Int) where {dim}
    f!(y, ae) = fλθ!(y, cv_u, X, ae, dir)
    result = ForwardDiff.jacobian!(result, f!, y, ae);
    return result
end

function fλθ!(f::Vector{T}, cv_u::CellVectorValues, X::Vector{Vec{dim,Float64}}, ae::Vector{T}, dir::Int) where {dim,T}
    e = basevec(Vec{dim}, dir)
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        u = function_value(cv_u, iqp, ae)
        xyz = spatial_coordinate(cv_u, iqp, X)
        z = xyz[dim]
        f[1] += (u ⋅ e*z) * dV 
    end
end

function integrate_fλθ_2!(ke::Matrix{T}, f::Vector{T}, cv_u::CellVectorValues, X::Vector{Vec{dim,Float64}}, ae::Vector{T}, dir::Int) where {dim,T}
    e = basevec(Vec{dim}, dir)
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        u = function_value(cv_u, iqp, ae)
        xyz = spatial_coordinate(cv_u, iqp, X)
        z = xyz[dim]
        f[1] += (u ⋅ e*z) * dV 
        for i in 1:getnbasefunctions(cv_u)
            Ni = shape_value(cv_u, iqp, i)
            ke[1,i] += (Ni ⋅ e*z) * dV
        end 
    end
end

function integrate_fμu!(result::DiffResult, y::Vector, fv_u::FaceVectorValues, ip_μ::Ferrite.Interpolation, X::Vector{Vec{dim,Float64}}, ae::Vector{Float64}) where {dim}
    f!(y, ae) = fμu!(y, fv_u, ip_μ, X, ae)
    result = ForwardDiff.jacobian!(result, f!, y, ae);
    return result
end

function fμu!(f::Vector{T}, cv_u::FaceVectorValues, ip_μ::Ferrite.Interpolation, X::Vector{Vec{dim,Float64}}, ae::Vector{T}) where {T,dim}

    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        u = function_value(cv_u, iqp, ae)
        xyz = spatial_coordinate(cv_u, iqp, X)
        z = xyz[dim]
        n = getnormal(cv_u, iqp)
        for i in 1:getnbasefunctions(ip_μ)
            δμ = shape_value(ip_μ, i, n, z)
            f[i] += (δμ ⋅ u) * dV 
        end
    end
    
end

function integrate_fμu_2!(ke::Matrix{T}, f::Vector{T}, cv_u::FaceVectorValues, ip_μ::Ferrite.Interpolation, X::Vector{Vec{dim,Float64}}, ae::Vector{T}) where {T,dim}

    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        u = function_value(cv_u, iqp, ae)
        xyz = spatial_coordinate(cv_u, iqp, X)
        z = xyz[dim]
        n = getnormal(cv_u, iqp)
        for i in 1:getnbasefunctions(ip_μ)
            δμ = shape_value(ip_μ, i, n, z)
            f[i] += (δμ ⋅ u) * dV 
            for j in 1:getnbasefunctions(cv_u)
                Nj = shape_value(cv_u, iqp, j)
                ke[i, j] += (δμ ⋅ Nj) * dV 
            end
        end
    end
    
end

function integrate_rhs!(f::Vector{Float64}, cv_u::CellVectorValues{dim}, material::MaterialModels.AbstractMaterial, state::Vector{<:MaterialModels.AbstractMaterialState}, X::Vector{Vec{dim,T}}, ∇uᴹ::Function) where {dim,T}
    for iqp in 1:getnquadpoints(cv_u)
        
        dV = getdetJdV(cv_u, iqp)
        
        x = spatial_coordinate(cv_u, iqp, X)
        εᴹ = symmetric(∇uᴹ(x))
        
        if dim == 2
            σ, C, state[iqp] = material_response(PlaneStrain(), material, εᴹ, state[iqp])
        else
            σ, C, state[iqp] = material_response(material, εᴹ, state[iqp])
        end
        
        for i in 1:getnbasefunctions(cv_u)
            ∇Ni = shape_gradient(cv_u, iqp, i)
            f[i] += (-∇Ni ⊡ σ ) * dV
        end 
    end

end

function integrate_fμ_ext!(f::Vector{T}, cv_u::FaceVectorValues, ip_μ::Ferrite.Interpolation, X::Vector{Vec{dim,Float64}}, ∇u, ∇w, ∇θ) where {dim,T}

    e = basevec.(Vec{dim}, ntuple(i->i, dim))
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        xyz = spatial_coordinate(cv_u, iqp, X)
        xₚ  = Vec{dim-1}(i -> xyz[i])
        z = xyz[dim]
        n = getnormal(cv_u, iqp)
        for i in 1:getnbasefunctions(ip_μ)
            δμ = shape_value(ip_μ, i, n, z)
            δμₚ = Vec{dim-1}(i -> δμ[i])
            δμ₃ = δμ[dim]

            f[i] -=  (δμₚ ⋅ (∇u⋅xₚ))   * dV 
            f[i] +=  (δμₚ ⋅ (z*∇θ⋅xₚ)) * dV
            f[i] -=  (δμ₃ * (∇w⋅xₚ))   * dV 
            
        end
    end  
end

function integrate_checks(val1::Vector{Float64}, val2::Vector{Float64}, cv_u::CellVectorValues, X::Vector{Vec{dim,Float64}}) where {dim}

    for iqp in 1:getnquadpoints(cv_u)
 
        dV = getdetJdV(cv_u, iqp)
        xyz = spatial_coordinate(cv_u, iqp, X)
        xₚ  = Vec{dim-1}(i -> xyz[i])
        z = xyz[dim]
        
        val1[1] += (xₚ[1]) * dV
        val2[1] += (z) * dV
    end

end

function u◫_operator(cv_u::CellVectorValues{dim}, ae::Vector{T}) where {dim,T}
    e = basevec(Vec{dim,Float64})
    Î = (e[1]⊗e[1]) + (e[2]⊗e[2])
    u◫ = zero(Vec{dim,T})
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        u = function_value(cv_u, iqp, ae)
        u◫ += (Î ⋅ u) * dV 
    end
    return u◫
end

function w◫_operator(cv_u::CellVectorValues{dim}, ae::Vector{T}) where {dim,T}
    e = basevec(Vec{dim,Float64})
    w = zero(T)
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        u = function_value(cv_u, iqp, ae)
        w += (e[3] ⋅ u) * dV 
    end
    return w
end

function θ◫_operator(cv_u::CellVectorValues{dim}, ae::Vector{T}, x::Vector{Vec{dim,T}}) where {dim,T}
    e = basevec(Vec{dim,Float64})
    Î = (e[1]⊗e[1]) + (e[2]⊗e[2])
    θ◫ = zero(Vec{dim,T})
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        θ = function_value(cv_u, iqp, ae)
        _,_,z = spatial_coordinate(cv_u, iqp, x)
        θ◫ += (z*Î ⋅ θ) * dV 
    end
    return θ◫
end

function h◫_operator(cv_u::CellVectorValues{dim}, ae::Vector{T}) where {dim,T}
    e = basevec(Vec{dim,Float64})
    Î = (e[1]⊗e[1]) + (e[2]⊗e[2])
    h◫ = zero(Tensor{2,dim,T})
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        ∇u = function_gradient(cv_u, iqp, ae)
        h◫ += (Î ⋅ ∇u ⋅ Î ) * dV 
    end
    return h◫
end

function g◫_operator(cv_u::CellVectorValues{dim}, ae::Vector{T}) where {dim,T}
    e = basevec(Vec{dim,Float64})
    Î = (e[1]⊗e[1]) + (e[2]⊗e[2])
    g◫ = zero(Vec{dim,T})
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        ∇u = function_gradient(cv_u, iqp, ae)
        g◫ += Î ⋅ (e[3] ⋅ ∇u) * dV 
    end
    return g◫
end

function κ◫_operator(cv_u::CellVectorValues{dim}, ae::Vector{T}, x::Vector{Vec{dim,T}}) where {dim,T}
    e = basevec(Vec{dim,Float64})
    Î = (e[1]⊗e[1]) + (e[2]⊗e[2])
    κ◫ = zero(Tensor{2,dim,T})
    for iqp in 1:getnquadpoints(cv_u)
        dV = getdetJdV(cv_u, iqp)
        ∇θ = function_gradient(cv_u, iqp, ae)
        _,_,z = spatial_coordinate(cv_u, iqp, x)
        κ◫ += (z*Î ⋅ ∇θ ⋅ Î ) * dV 
    end
    return κ◫
end
