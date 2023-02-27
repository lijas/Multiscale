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

function integrate_!(
    ke_uu::Matrix{Float64}, 
    ke_ξu::Matrix{Float64}, 
    ke_ξξ::Matrix{Float64}, 
    fu::Vector{Float64}, 
    fξ::Vector{Float64}, 
    coords::Vector{Vec{dim,Float64}}, 
    cv_u::CellVectorValues{dim}, 
    material::MaterialModels.AbstractMaterial, 
    state::Vector{<:MaterialModels.AbstractMaterialState}, 
    ae::Vector{T}) where {dim,T}
    
    e1, e2, e3 = basevec(Vec{dim,Float64})
    Î = (e1⊗e1) + (e2⊗e2)
    x̄ = zero(Vec{dim,Float64})

    δεκ = zeros(SymmetricTensor{2,3}, 3)
    shape_function_kappa = [
        SymmetricTensor{2,3,Float64,6}((1.0,0.0,0.0, 0.0,0.0, 0.0)),
        SymmetricTensor{2,3,Float64,6}((0.0,1.0,0.0, 0.0,0.0, 0.0)),
        SymmetricTensor{2,3,Float64,6}((0.0,0.0,0.0, 1.0,0.0, 0.0))
    ]

    udofs = 1:getnbasefunctions(cv_u)
    κdofs = (1:3) .+ getnquadpoints(cv_u)

    for iqp in 1:getnquadpoints(cv_u)
        
        dV = getdetJdV(cv_u, iqp)
        x = spatial_coordinate(cv_u, iqp, coords)
        ∇u = function_gradient(cv_u, iqp, ae)
        ε = symmetric(∇u)
        
        x̂ = Î ⋅ x

        σ, C, state[iqp] = material_response(material, ε, state[iqp])
        for i in 1:3
            Nκ = shape_function_kappa[i]
            ∇uκ = (Î ⋅ Nκ ⋅ (x̂ - x̄)) ⊗ e3
            δεκ[i] = symmetric(∇uκ)
        end

        for (i,I) in pairs(udofs)
            δεi = (shape_gradient(cv_u, iqp, i))
            fu[i] += (σ ⊡ δεi) * dV
            for (j,J) in pairs(udofs)
                Δεj = (shape_gradient(cv_u, iqp, j))
                ke_uu[i,j] += (Δεj ⊡ C ⊡ δεi) * dV
            end

            #=for (j,J) in pairs(κdofs)
                Δεκj = δεκ[j]
                ke_uξ[i,j] += (Δεκj ⊡ C ⊡ δεi) * dV
            end=#
        end 

        
        for (i,I) in pairs(κdofs)
            δεκi = δεκ[i]
            fξ[i] += (σ ⊡ δεκi) * dV
            for (j,J) in pairs(udofs)
                Δεj = symmetric(shape_gradient(cv_u, iqp, j))
                ke_ξu[i,j] += (Δεj ⊡ C ⊡ δεκi) * dV
            end

            for (j,J) in pairs(κdofs)
                δεκj = δεκ[j]
                ke_ξξ[i,j] += (δεκj ⊡ C ⊡ δεκi) * dV
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

function integrate_rhs!(fu::Vector{Float64}, fξ::Vector{Float64}, cv_u::CellVectorValues{dim}, material::MaterialModels.AbstractMaterial, state::Vector{<:MaterialModels.AbstractMaterialState}, X::Vector{Vec{dim,T}}, ∇uᴹ::Function) where {dim,T}
    e1, e2, e3 = basevec(Vec{dim,Float64})
    Î = (e1⊗e1) + (e2⊗e2)
    x̄ = zero(Vec{dim,Float64})

    δεξ = zeros(SymmetricTensor{2,3}, 3)
    shape_function_kappa = [
        SymmetricTensor{2,3,Float64,6}((1.0,0.0,0.0, 0.0,0.0, 0.0)),
        SymmetricTensor{2,3,Float64,6}((0.0,1.0,0.0, 0.0,0.0, 0.0)),
        SymmetricTensor{2,3,Float64,6}((0.0,0.0,0.0, 1.0,0.0, 0.0))
    ]
    for iqp in 1:getnquadpoints(cv_u)
        
        dV = getdetJdV(cv_u, iqp)
        
        x = spatial_coordinate(cv_u, iqp, X)
        εᴹ = symmetric(∇uᴹ(x))

        σ, C, state[iqp] = material_response(material, εᴹ, state[iqp])
        
        x̂ = Î ⋅ x

        for i in 1:3
            Nκ = shape_function_kappa[i]
            ∇uκ = (Î ⋅ Nκ ⋅ (x̂ - x̄)) ⊗ e3
            δεξ[i] = symmetric(∇uκ)
        end

        for i in 1:getnbasefunctions(cv_u)
            ∇Ni = shape_gradient(cv_u, iqp, i)
            fu[i] += (-∇Ni ⊡ σ ) * dV
        end 

        for i in 1:3
            ∇Nξi = δεξ[i]
            fξ[i] += (-∇Nξi ⊡ σ ) * dV
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

function integrate_volume_check(cv_u::CellVectorValues, X::Vector{Vec{dim,Float64}}) where {dim}

    val1 = 0.0
    val2 = 0.0
    for iqp in 1:getnquadpoints(cv_u)
 
        dV = getdetJdV(cv_u, iqp)
        xyz = spatial_coordinate(cv_u, iqp, X)
        xₚ  = Vec{dim-1}(i -> xyz[i])
        z = xyz[dim]
        
        val1 += (xₚ[1]) * dV
        val2 += (z) * dV
    end
    return val1, val2
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
