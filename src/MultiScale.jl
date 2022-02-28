
module MultiScale

using MaterialModels
using Ferrite
using ForwardDiff
using DiffResults: DiffResult
using SparseArrays

include("integrals.jl")
include("extra_materials.jl")

#=struct ShellPartFe2

    cellset::Vector{Int}
    qp_rve_pointers::Vector{Int} # qp --> RVE geometry
    rves::Vector{RVE} # A list of RVEs

end=#


struct RVESubPartState{MS<:MaterialModels.AbstractMaterialState}
    materialstates::Vector{Vector{MS}}
end

struct TractionInterpolation{dim} end

function Ferrite.shape_value(::TractionInterpolation{3}, i::Int, n::Vec{3,Float64}, z::T) where T
    dir1 = all( abs(n[1]) ≈ 1.0 && n[2] ≈ 0.0 && n[3] ≈ 0.0) ? true : false
    #@show dir1, n
    i==1 && return dir1 ? Vec((1.0, 0.0, 0.0)) : Vec((0.0, 0.0, 0.0))
    i==2 && return dir1 ? Vec((0.0, 0.0, 0.0)) : Vec((1.0, 0.0, 0.0))
    i==3 && return dir1 ? Vec((0.0, 1.0, 0.0)) : Vec((0.0, 0.0, 0.0))
    i==4 && return dir1 ? Vec((0.0, 0.0, 0.0)) : Vec((0.0, 1.0, 0.0))
    i==5 && return dir1 ? Vec(( z , 0.0, 0.0)) : Vec((0.0, 0.0, 0.0))
    i==6 && return dir1 ? Vec((0.0, 0.0, 0.0)) : Vec(( z , 0.0, 0.0))
    i==7 && return dir1 ? Vec((0.0,  z , 0.0)) : Vec((0.0, 0.0, 0.0))
    i==8 && return dir1 ? Vec((0.0, 0.0, 0.0)) : Vec((0.0,  z , 0.0))
    i==9 && return dir1 ? Vec((0.0, 0.0, 1.0)) : Vec((0.0, 0.0, 0.0))
   i==10 && return dir1 ? Vec((0.0, 0.0, 0.0)) : Vec((0.0, 0.0, 1.0))
    error("Wrong iii")
end

function Ferrite.shape_value(::TractionInterpolation{2}, i::Int, ::Vec{2,Float64}, z::T) where T
    i==1 && return (1.0, 0.0) |> Vec{2,T}
    i==2 && return (0.0, 1.0) |> Vec{2,T}
    i==3 && return (z, 0.0)   |> Vec{2,T}
    #i==4 && return (0.0, z)   |> Vec{2,T}
    error("Wrong iii")
end

Ferrite.getnbasefunctions(::TractionInterpolation{2}) = 3
Ferrite.getnbasefunctions(::TractionInterpolation{3}) = 10

Base.@propagate_inbounds function disassemble!(ue::AbstractVector, u::AbstractVector, dofs::AbstractVector{Int})
    Base.@boundscheck checkbounds(u, dofs)
    # @inbounds for i in eachindex(ue, dofs) # Slow on Julia 1.6 (JuliaLang/julia#40267)
    Base.@inbounds for i in eachindex(ue)
        ue[i] = u[dofs[i]]
    end
    return ue
end

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

@Base.kwdef struct RVESubPart{M<:MaterialModels.AbstractMaterial}
    material::M
    cellset::Vector{Int}
end

struct RVECache{dim}
    udofs::Vector{Int}
    X::Vector{Vec{dim,Float64}}
    ae::Vector{Float64}
    fu::Vector{Float64}
    ke::Matrix{Float64}
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
    fλ = zeros(Float64, 1)

    diffresult_u = DiffResults.JacobianResult(fu, ae)
    diffresult_μ = DiffResults.JacobianResult(fμ, ae)
    diffresult_λ = DiffResults.JacobianResult(fλ, ae)

    return RVECache{dim}(udofs, X, ae, fu ,ke, fμ, fλ, diffresult_u, diffresult_μ, diffresult_λ)
end

@enum BCType WEAK_PERIODIC STRONG_PERIODIC

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
    L◫::NTuple{dim,Float64}
    Ω◫::Float64
    A◫::Float64
    I◫::Float64

    #
    nudofs::Int
    nμdofs::Int
    nλdofs::Int

    #
    BC_TYPE::BCType
    
end

function rvesize(grid::Grid{dim}; dir=d) where dim
    @assert(dim >= dir)
    maxx = -Inf
    minx = +Inf
    for node in grid.nodes
        maxx = max(maxx, node.x[dir])
        minx = min(minx, node.x[dir])
    end
    return maxx - minx

end

function RVE(; grid::Grid{dim}, parts::Vector{RVESubPart}, BC_TYPE::BCType = WEAK_PERIODIC) where dim

    #Get size of rve
    side_length = ntuple( d -> rvesize(grid; dir = d), dim)

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

    #ConstraintHandler
    ch = ConstraintHandler(dh)
    if BC_TYPE == STRONG_PERIODIC
        add!(ch, Dirichlet(:u, Γ⁺, (x,t) -> (0.0,0.0,0.0), [1,2,3]))
    end    
    close!(ch)

    #Element
    ip_u = Lagrange{dim,RefCube,1}()
    qr   = QuadratureRule{dim,RefCube}(2)
    qr_face   = QuadratureRule{dim-1,RefCube}(2)

    cv_u = CellVectorValues(qr, ip_u)
    fv_u = FaceVectorValues(qr_face, ip_u)

    #
    nudofs = ndofs_per_cell(dh)
    nμdofs = getnbasefunctions(TractionInterpolation{dim}())
    nλdofs = dim*2 - 1
    
    cache = RVECache(dim, nudofs, nμdofs, nλdofs)
    matrices = Matrices(dh, nμdofs, nλdofs)

    #
    h = side_length[dim]
    Ω◫ = prod(side_length)
    A◫ = Ω◫ / h
    I◫ = A◫*h^3/12

    return RVE(grid, dh, ch, parts, cache, matrices, cv_u, fv_u, side_length, Ω◫, A◫, I◫, nudofs, nμdofs, nλdofs, BC_TYPE)
end


struct State
    a::Vector{Float64}
    partstates::Vector{RVESubPartState}
end

function State(rve::RVE)
    
    partstates = RVESubPartState[]
    for part in rve.parts
        MS = typeof( initial_material_state(part.material) )
        materialstates = Vector{<:AbstractMaterialState}[]
        for (localid, cellid) in enumerate(part.cellset)
            qpstates = [initial_material_state(part.material) for _ in 1:getnquadpoints(rve.cv_u)]
            push!(materialstates, qpstates)
        end
        push!(partstates, RVESubPartState{MS}(materialstates))
    end

    _ndofs = ndofs(rve.dh) + rve.nμdofs + rve.nλdofs

    return State(zeros(Float64, _ndofs), partstates)
end

function State(rve::RVE, partstates::Vector{RVESubPartState}) 
    _ndofs = ndofs(rve.dh) + rve.nμdofs + rve.nλdofs
    return State(zeros(Float64, _ndofs), partstates)
end

function solve_rve(rve::RVE, macroscale::MacroParameters, state::State)

    assemble_volume!(rve, state)
    
    if rve.BC_TYPE == WEAK_PERIODIC
        assemble_face!(rve, macroscale, state.a)
    end

    a = solve_it!(rve, state)

    return a
end

function assemble_volume!(rve::RVE, state::State)

    fill!(rve.matrices.Kuu, 0.0)
    fill!(rve.matrices.Kλu, 0.0)
    fill!(rve.matrices.Kμu, 0.0)
    fill!(rve.matrices.fuu, 0.0)

    for (partid, part) in enumerate(rve.parts)
        material = part.material
        cellset = part.cellset
        matstates = state.partstates[partid].materialstates

        _assemble!(rve, material, matstates, cellset, state.a)
    end

end


function _assemble!(rve::RVE{dim}, material::AbstractMaterial, materialstates::Vector{Vector{MS}}, cellset::Vector{Int}, a::Vector{Float64}) where {dim, MS<:AbstractMaterialState}

    (; udofs, X, ae, fu, fλ, ke) = rve.cache
    (; grid, dh, cv_u)    = rve
    (; matrices) = rve
    (; diffresult_u, diffresult_λ) = rve.cache
    (; A◫, Ω◫, I◫) = rve

    assembler_u = start_assemble(matrices.Kuu, matrices.fuu, fillzero=false)

    #a = zeros(Float64, ndofs(dh)) # TODO: where to put this?
    #materialstates = [[initial_material_state(material) for i in 1:getnquadpoints(cv_u)] for _ in 1:getncells(grid)]

    for (localid, cellid) in enumerate(cellset)
        fill!(fu, 0.0)
        fill!(ke, 0.0)

        celldofs!(udofs, dh, cellid)
        getcoordinates!(X, grid, cellid)
        disassemble!(ae, a, udofs)

        mstates = materialstates[localid]

        reinit!(cv_u, X)

        #diffresult_u = integrate_fuu2!(diffresult_u, fu, cv_u, material, mstates, ae);
        #fu = 1/A◫ * DiffResults.value(diffresult_u)
        #ke = 1/A◫ * DiffResults.jacobian(diffresult_u)
        
        integrate_fuu2!(ke, fu, cv_u, material, mstates, ae);
        assemble!(assembler_u, udofs, ke, fu)

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

function assemble_face!(rve::RVE{dim}, macroscale::MacroParameters, a::Vector{Float64}) where dim

    (; udofs, X, ae, fμ, diffresult_μ) = rve.cache
    (; grid, dh, fv_u)    = rve
    (; matrices) = rve
    (; diffresult_μ) = rve.cache
    (; A◫) = rve

    μdofs = collect(1:rve.nμdofs)

    Γ⁺ = union( getfaceset(grid, "Γ⁺") ) |> collect
    Γ⁻ = union( getfaceset(grid, "Γ⁻") ) |> collect
    
    #a = zeros(Float64, ndofs(dh)) # TODO: where to put this?

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

function solve_it!(rve::RVE, state::State)

    if rve.BC_TYPE == STRONG_PERIODIC
        _solve_it_strong_periodic(rve, state)
    elseif rve.BC_TYPE == WEAK_PERIODIC
        _solve_it_weak_periodic(rve, state)
    else
        error("WRONG BC_TYPE")
    end

end

function _solve_it_weak_periodic(rve::RVE, state::State)

    (; dh, matrices)   = rve
    (; nμdofs, nλdofs) = rve

    K = vcat(hcat(matrices.Kuu, matrices.Kμu', matrices.Kλu'),
        hcat(matrices.Kμu, zeros(Float64, nμdofs, nμdofs), zeros(Float64,nμdofs,nλdofs)),
        hcat(matrices.Kλu, zeros(Float64, nλdofs, nμdofs), zeros(Float64,nλdofs,nλdofs)))

    fext_u = zeros(Float64, ndofs(dh))
    f = vcat(fext_u, matrices.fext_μ, matrices.fext_λ)

    state.a .= K\f

    #μdofs = (1:nμdofs) .+ ndofs(dh)
    #λdofs = (1:nλdofs) .+ ndofs(dh) .+ nμdofs

    #μ = a[μdofs]
    #λ = a[λdofs]
end


function _solve_it_strong_periodic(rve::RVE, state::State)

    (; dh, matrices)   = rve
    (; nμdofs, nλdofs) = rve

    K = vcat(hcat(matrices.Kuu, matrices.Kμu', matrices.Kλu'),
        hcat(matrices.Kμu, zeros(Float64, nμdofs, nμdofs), zeros(Float64,nμdofs,nλdofs)),
        hcat(matrices.Kλu, zeros(Float64, nλdofs, nμdofs), zeros(Float64,nλdofs,nλdofs)))

    fext_u = zeros(Float64, ndofs(dh))
    f = vcat(fext_u, matrices.fext_μ, matrices.fext_λ)

    apply!(K, f, rve.ch)
    state.a .= K\f
    apply!(state.a, rve.ch)

    #μdofs = (1:nμdofs) .+ ndofs(dh)
    #λdofs = (1:nλdofs) .+ ndofs(dh) .+ nμdofs

    #μ = a[μdofs]
    #λ = a[λdofs]
end

include("response.jl")

export MacroParameters
export RVESubPartState
export State
end

