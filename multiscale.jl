
module MultiScale

using MaterialModels
using Ferrite
using ForwardDiff
using DiffResults: DiffResult
using SparseArrays

#=struct ShellPartFe2

    cellset::Vector{Int}
    qp_rve_pointers::Vector{Int} # qp --> RVE geometry
    rves::Vector{RVE} # A list of RVEs

end=#

struct Matrices
    Kuu::SparseMatrixCSC{Float64,Int}
    Kμu::Matrix{Float64}
    Kλu::Matrix{Float64}

    fuu::Vector{Float64}
    fμu::Vector{Float64}
    fλu::Vector{Float64}

    fext_μ::Vector{Float64}
    fext_λ::Vector{Float64}
end

function Matrices(dh::DofHandler, nμ::Int, nλ::Int)
    Kuu = create_sparsity_pattern(dh)
    return Matrices(Kuu, spzeros(nμ, ndofs(dh)), spzeros(nλ, ndofs(dh)), zeros(Float64, ndofs(dh)), zeros(Float64, ndofs(dh)), zeros(Float64, ndofs(dh)), zeros(Float64, nμ), zeros(Float64, nλ))
end


struct MacroParameters{dim,M}
    ∇u::Tensor{2,dim,Float64,M}  
    ∇w::Vec{dim,Float64} 
    ∇θ::Tensor{2,dim,Float64,M} 
    u::Vec{dim,Float64} 
    w::Float64 
    θ::Vec{dim,Float64} 
    #dθ::Tensor{N,2,Float64}
end

function MacroParameters{N}(;
    ∇u::Tensor{2,N,Float64} = zero(Tensor{2,N,Float64,4}), 
    ∇w::Vec{N,Float64} = zero(Vec{N, Float64}), 
    ∇θ::Tensor{2,N,Float64}  = zero(Tensor{2,N,Float64,4}), 
    u::Vec{N,Float64}  = zero(Vec{N, Float64}), 
    w::Float64 = 0.0,
    θ::Vec{N,Float64} = zero(Vec{N, Float64})
    ) where N

    return MacroParameters{N,N*N}(∇u, ∇w, ∇θ, u, w, θ)

end

struct RVESubPart{M<:MaterialModels.AbstractMaterial}
    material::M
    cellset::Vector{Int}
end

struct RVECache{dim}
    udofs::Vector{Int}
    X::Vector{Vec{dim,Float64}}
    ae::Vector{Int}
    fu::Vector{Float64}
    ke::Vector{Float64}
    fμ::Vector{Float64}
    fλ::Vector{Float64}
    diffresult_u::DiffResult
    diffresult_μ::DiffResult
    diffresult_λ::DiffResult
end

function RVECache(dim::Int, nudofs, nμdofs, nλdofs)

    udofs = zeros(Int, nudofs)
    X = zeros(Vec{dim,Float64}, nudofs÷dim)
    ae = zeros(Float64, nudofs)

    ae = zeros(Float64, nudofs)
    fu = zeros(Float64, nudofs)
    ke = zeros(Float64, nudofs, nudofs)
    fμ = zeros(Float64, nμdofs)
    fλ = zeros(Float64, nλdofs)

    diffresult_u = DiffResults.JacobianResult(fu, ae)
    diffresult_μ = DiffResults.JacobianResult(fμ, ae)
    diffresult_λ = DiffResults.JacobianResult(fλ, ae)

    return RVECache{dim}(udofs, X, ae, ae, fu ,ke, fμ, fλ, diffresult_u, diffresult_μ, diffresult_λ)
end

struct RVE{dim}

    grid::Grid{dim}
    dh::DofHandler{dim}
    ch::ConstraintHandler 
    parts::Vector{RVESubPart}

    cache::RVECache

    #System matrices
    matrices::Matrices

    #Element
    cv_u::CellValues
    fv_u::FaceValues

    #
    Ω◫::Float64
    A◫::Float64
end

function RVE(grid::Grid)
    
    xcoords = getindex.(getproperty.(grid.nodes, :x), 1)
    ycoords = getindex.(getproperty.(grid.nodes, :x), 2)    
    Lx◫ = maximum(xcoords) - minimum(xcoords)
    Ly◫ = maximum(ycoords) - minimum(ycoords)

    @assert(Lx◫ == Ly◫)
    L◫ = Lx◫

    #Periodic boundaries
    Γ⁺ = getfaceset(grid, "right")
    Γ⁻ = getfaceset(grid, "left")
    if dim == 3
        union!(Γ⁺, getfaceset(grid, "front"))
        union!(Γ⁻, getfaceset(grid, "back"))
    end
    addfaceset!(grid, "Γ⁺", Γ⁺)
    addfaceset!(grid, "Γ⁻", Γ⁻)

    #Dofhandler
    dh = DofHandler(grid)
    push!(dh, :u, dim)
    close!(dh)

    #Element
    ip_u = Lagrange{dim,RefCube,1}()
    qr   = QuadratureRule{dim,RefCube}(2)
    qr_face   = QuadratureRule{dim-1,RefCube}(2)

    cv_u = CellVectorValues(qr, ip_u)
    fv_u = FaceVectorValues(qr_face, ip_u)

    #
    nudofs = ndofs_per_cell(dh)
    nμdofs = getnbasefunctions(TractionInterpolation{dim}())
    μdofs = collect(1:nμdofs)
    nλdofs = dim*2 - 1
    
    cache = RVECache(dim, nudofs, nμdofs, nλdofs)

    #
    A◫ = L◫^(dim-1)

    return RVE(grid, dh, ch, parts, cache, matrices, cv_u, fv_u, Ω◫, A◫)
end

function sovle_rve(rve::RVE, macroscale::MacroParameters)


    assemble_volume!(rve)
    assemble_face!(rve)

    solve_it()

end

function assemble_volume!(rve::RVE)

    for part in RVE.parts
        material = part.material
        cellset = part.cellset

        _assemble!(dh, material, cellset)
    end

end


function _assemble!(rve::RVE{dim}, material::AbstractMaterial, cellset::Vector{Int}) where dim

    (; udofs, X, ae) = rve.cache
    (; grid, dh, cv_u)    = rve
    (; matrices) = rve
    (; diffresult_u, diffresult_μ, diffresult_λ) = rve.cache

    assembler_u = start_assemble(matrices.Kuu, matrices.fuu)


    for (localid, cellid) in enumerate(cellset)
        fill!(fu, 0.0)
        fill!(ke, 0.0)

        celldofs!(udofs, dh, cellid)
        getcoordinates!(X, grid, cellid)
        disassemble!(ae, a, udofs)

        reinit!(cv_u, X)

        diffresult_u = integrate_fuu!(diffresult_u, fu, cv_u, material, ae);
        #@timeit "Integrate2 u" integrate_fuu2!(ke, fu, cv_u, C, ae);

        fe = 1/A◫ * DiffResults.value(diffresult_u)
        ke = 1/A◫ * DiffResults.jacobian(diffresult_u)
        assemble!(assembler_u, udofs, ke, fe)

        #---
        for d in 1:dim
            fill!(fλ, 0.0)
            diffresult_λ = integrate_fλu!(diffresult_λ, fλ, cv_u, ae, d); 
            Ke = DiffResults.jacobian(diffresult_λ)
            matrices.Kλu[[d], udofs] += -Ke * (1/Ω◫)
        end 

        for d in 1:dim-1
            fill!(fλ, 0.0)
            diffresult_λ = integrate_fλθ!(diffresult_λ, fλ, cv_u, X, ae, d); 
            Ke = DiffResults.jacobian(diffresult_λ)
            matrices.Kλu[[dim+d], udofs] += Ke * (1/I◫)
        end
    end


end

function assemble_face!(rve::RVE{dim}) where dim

    (; udofs, X, ae) = rve.cache
    (; grid, dh, fv_u)    = rve
    (; matrices) = rve

    Γ⁺ = union( getfaceset(grid, "Γ⁺") ) |> collect
    Γ⁻ = union( getfaceset(grid, "Γ⁻") ) |> collect
    
    for (cellid, fidx) in Γ⁺
        celldofs!(udofs, dh, cellid)
        getcoordinates!(X, dh.grid, cellid)
        disassemble!(ae, a, udofs)

        reinit!(fv_u, X, fidx)

        fill!(fμ, 0.0)
        diffresult_μ = integrate_fμu!(diffresult_μ, fμ, fv_u, X, ae);

        #fe = DiffResults.value(diffresult_μ)
        Ke = DiffResults.jacobian(diffresult_μ)
        matrices.Kμu[μdofs, udofs] += -Ke  * (1/A◫) 

        fill!(fμ, 0.0)
        integrate_fμ_ext!(fμ, fv_u, X, macroscale.∇u, macroscale.∇w, macroscale.∇θ);
        matrices.fext_μ[μdofs] += fμ * (1/A◫)
    end

    for (cellid, fidx) in Γ⁻
        
        celldofs!(udofs, dh, cellid)
        getcoordinates!(X, dh.grid, cellid)        
        disassemble!(ae, a, udofs)
        
        reinit!(fv_u, X, fidx)
        
        fill!(fμ, 0.0)
        diffresult_μ = integrate_fμu!(diffresult_μ, fμ, fv_u, X, ae);

        #fe = DiffResults.value(diffresult_μ)
        Ke = DiffResults.jacobian(diffresult_μ)
        matrices.Kμu[μdofs, udofs] += Ke  * (1/A◫) 

        fill!(fμ, 0.0)
        integrate_fμ_ext!(fμ, fv_u, X, macroscale.∇u, macroscale.∇w, macroscale.∇θ);
        matrices.fext_μ[μdofs] += -fμ * (1/A◫)
    end

    for d in 1:dim-1
        matrices.fext_λ[d] +=  -macroscale.u[d]
        matrices.fext_λ[dim+d] +=  -macroscale.θ[d] #-(macroscale.ϕ - macroscale.γ/2)
    end
    matrices.fext_λ[dim] +=  -macroscale.w

end

function solve_hit()

    K, f = rve.K, rve.f

    K = vcat(hcat(matrices.Kuu, matrices.Kμu', matrices.Kλu'),
    hcat(matrices.Kμu, zeros(Float64, nμdofs, nμdofs), zeros(Float64,nμdofs,nλdofs)),
    hcat(matrices.Kλu, zeros(Float64, nλdofs, nμdofs), zeros(Float64,nλdofs,nλdofs)))

    #display(K)
    fext_u = zeros(Float64, ndofs(dh))
    f = vcat(fext_u, matrices.fext_μ, matrices.fext_λ)


    @show norm(matrices.Kuu)
    @show norm(matrices.Kμu)
    @show norm(matrices.Kλu)
    @show norm(matrices.fext_μ)

    #apply!(K,f,ch)
    a = K\f
    #apply!(a,ch)
    μdofs = (1:nμdofs) .+ ndofs(dh)
    λdofs = (1:nλdofs) .+ ndofs(dh) .+ nμdofs

    μ = a[μdofs]
    λ = a[λdofs]
end

end

