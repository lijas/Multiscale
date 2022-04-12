
module MultiScale

using MaterialModels
using Ferrite
#using FerriteGmsh
using Tensors
using ForwardDiff
using DiffResults: DiffResult
using SparseArrays
using Random, Distributions
using Plots: plot, plot!, scatter3d!, plot3d!

export WEAK_PERIODIC, STRONG_PERIODIC, STRONG_PERIODIC_WITH_PAIRS, DIRICHLET, RELAXED_DIRICHLET

include("integrals.jl")
include("extra_materials.jl")
include("sampledomain.jl")
#include("gmshdomain.jl")

#=struct ShellPartFe2

    cellset::Vector{Int}
    qp_rve_pointers::Vector{Int} # qp --> RVE geometry
    rves::Vector{RVE} # A list of RVEs

end=#

increase_dim(A::Tensor{1,dim,T}) where {dim, T} = Tensor{1,dim+1}(i->(i <= dim ? A[i] : zero(T)))
increase_dim(A::Tensor{2,dim,T}) where {dim, T} = Tensor{2,dim+1}((i,j)->(i <= dim && j <= dim ? A[i,j] : zero(T)))


struct RVESubPartState{MS<:MaterialModels.AbstractMaterialState}
    materialstates::Vector{Vector{MS}}
end

struct TractionInterpolation{dim} <: Ferrite.Interpolation{dim,RefCube,1} end

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

struct RelaxedDirichletInterpolation{dim} <: Ferrite.Interpolation{dim,RefCube,0} end

function Ferrite.shape_value(::RelaxedDirichletInterpolation{3}, i::Int, n::Vec{3,Float64}, z::T) where T
    dir1 = all( abs(n[1]) ≈ 1.0 && n[2] ≈ 0.0 && n[3] ≈ 0.0) ? true : false
    i==1 && return dir1 ? Vec((0.0, 0.0, 1.0)) : Vec((0.0, 0.0, 0.0))
    i==2 && return dir1 ? Vec((0.0, 0.0, 0.0)) : Vec((0.0, 0.0, 1.0))
    error("Wrong iii")
end

function Ferrite.shape_value(::RelaxedDirichletInterpolation{2}, i::Int, ::Vec{2,Float64}, z::T) where T
    i==1 && return (0.0, 1.0) |> Vec{2,T}
    error("Wrong iii")
end

Ferrite.getnbasefunctions(::RelaxedDirichletInterpolation{2}) = 1
Ferrite.getnbasefunctions(::RelaxedDirichletInterpolation{3}) = 2

Base.@propagate_inbounds function disassemble!(ue::AbstractVector, u::AbstractVector, dofs::AbstractVector{Int})
    Base.@boundscheck checkbounds(u, dofs)
    # @inbounds for i in eachindex(ue, dofs) # Slow on Julia 1.6 (JuliaLang/julia#40267)
    Base.@inbounds for i in eachindex(ue)
        ue[i] = u[dofs[i]]
    end
    return ue
end

mutable struct Matrices
    Kuu::SparseMatrixCSC{Float64,Int}
    Kμu::Matrix{Float64}
    Kλu::Matrix{Float64}

    fuu::Vector{Float64}
    fμu::Vector{Float64}
    fλu::Vector{Float64}

    fext_μ::Vector{Float64}
    fext_λ::Vector{Float64}
end

function Matrices(dh::DofHandler, ch::ConstraintHandler, nμ::Int, nλ::Int)
    Kuu = spzeros(Float64, ndofs(dh), ndofs(dh)) #create_sparsity_pattern(dh, ch)
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

abstract type BCType end
struct WEAK_PERIODIC <: BCType end
struct STRONG_PERIODIC  <: BCType end
struct RELAXED_DIRICHLET  <: BCType end
struct DIRICHLET <: BCType end
struct STRONG_PERIODIC_WITH_PAIRS  <: BCType 
    pairs::Vector{Pair{Int,Int}}
end
@enum SolveStyle SOLVE_SCHUR SOLVE_FULL

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
    ip_μ::Union{Nothing,Ferrite.Interpolation} # for the δμ field

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
    SOLVE_STYLE::SolveStyle
    
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

function RVE(; grid::Grid{dim}, parts::Vector{RVESubPart}, BC_TYPE::BCType, SOLVE_STYLE::SolveStyle = SOLVE_FULL) where dim

    #Get size of rve
    side_length = ntuple( d -> rvesize(grid; dir = d), dim)

    #Dofhandler
    dh = DofHandler(grid)
    push!(dh, :u, dim)
    close!(dh)

    #ConstraintHandler
    ch = ConstraintHandler(dh)
    #close!(ch)

    celltype = getcelltype(grid)
    ip_u = Ferrite.default_interpolation(celltype)
    refshape = Ferrite.getrefshape(ip_u)
    @info "$refshape, $celltype, $ip_u"

    #Element
    qr   = QuadratureRule{dim,refshape}(3)
    qr_face   = QuadratureRule{dim-1,refshape}(3)

    cv_u = CellVectorValues(qr, ip_u)
    fv_u = FaceVectorValues(qr_face, ip_u)

    #
    nudofs = ndofs_per_cell(dh)
    ip_μ = nothing
    nμdofs = 0
    if BC_TYPE == WEAK_PERIODIC()
        ip_μ   = TractionInterpolation{dim}()
        nμdofs = getnbasefunctions(ip_μ)
        @assert( haskey(grid.facesets, "Γ⁺") )
        @assert( haskey(grid.facesets, "Γ⁻") )
        @assert( SOLVE_STYLE == SOLVE_FULL)
    elseif BC_TYPE == DIRICHLET()
        @assert( haskey(grid.facesets, "Γ⁺") )
        @assert( haskey(grid.facesets, "Γ⁻") )
    elseif BC_TYPE == STRONG_PERIODIC()
        @assert( haskey(grid.nodesets, "right") )
        @assert( haskey(grid.nodesets, "left") )
        if dim == 3
            @assert( haskey(grid.nodesets, "back") )
            @assert( haskey(grid.nodesets, "front") )
        end
    elseif BC_TYPE isa STRONG_PERIODIC_WITH_PAIRS
        @assert( false )
    elseif BC_TYPE == RELAXED_DIRICHLET()
        ip_μ = RelaxedDirichletInterpolation{dim}()
        nμdofs = getnbasefunctions(ip_μ)
        @assert( haskey(grid.facesets, "Γ⁺") )
        @assert( haskey(grid.facesets, "Γ⁻") )    
        @assert( SOLVE_STYLE == SOLVE_FULL)    
    else
        error("Wrong BCTYPE")
    end
    nλdofs = dim-1#*2 - 1
    
    cache = RVECache(dim, nudofs, nμdofs, nλdofs)
    matrices = Matrices(dh, ch, nμdofs, nλdofs)

    #
    h = side_length[dim]
    Ω◫ = prod(side_length)
    A◫ = Ω◫ / h
    I◫ = A◫*h^3/12

    return RVE(grid, dh, ch, parts, cache, matrices, cv_u, fv_u, ip_μ, side_length, Ω◫, A◫, I◫, nudofs, nμdofs, nλdofs, BC_TYPE, SOLVE_STYLE)
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

function solve_rve(rve::RVE{dim}, macroscale::MacroParameters, state::State) where dim
    @info "applying macroscale"
    _apply_macroscale!(rve, macroscale, state)

    @info "assembling volume"
    _assemble_volume!(rve, macroscale, state)

    @info "Solving it"
    a = solve_it!(rve, state)
    return a
end

function _apply_macroscale!(rve::RVE{dim}, macroscale::MacroParameters, state::State) where dim

    ∇u = increase_dim(macroscale.∇u)
    ∇θ = increase_dim(macroscale.∇θ)
    ∇w = increase_dim(macroscale.∇w)

    e₃ = basevec(Vec{dim}, dim)

    if rve.BC_TYPE == WEAK_PERIODIC()
        @info "Assemble face"
        assemble_face!(rve, macroscale, state.a)

        
        dbc = Ferrite.Dirichlet(
            :u,
            getnodeset(rve.grid, "cornerset"),
            (x, t) -> zero(Vec{dim,Float64}),
            1:dim#(dim-1)
        )
        add!(rve.ch, dbc)

    elseif rve.BC_TYPE == DIRICHLET()
        x̄ = zero(Vec{dim})

        facesnames = ["right", "left"]
        if dim ==3
            append!(facesnames, ["front", "back"])
        end

        dbc = Ferrite.Dirichlet(
            :u,
            union(getfaceset.(Ref(rve.grid), facesnames)...),
            (x, t) ->  (∇u⋅(x-x̄) - x[dim]*∇θ⋅(x-x̄) + ∇w⋅(x-x̄) * e₃),#[1:dim-1],
            1:dim#(dim-1)
        )
        add!(rve.ch, dbc)

    elseif rve.BC_TYPE == RELAXED_DIRICHLET()
        assemble_face!(rve, macroscale, state.a)
        x̄ = zero(Vec{dim})

        facesnames = ["right", "left"]
        if dim ==3
            append!(facesnames, ["front", "back"])
        end

        dbc = Ferrite.Dirichlet(
            :u,
            union(getfaceset.(Ref(rve.grid), facesnames)...),
            (x, t) ->  (∇u⋅(x-x̄) - x[dim]*∇θ⋅(x-x̄))[1:dim-1],
            1:(dim-1)
        )
        add!(rve.ch, dbc)

        dbc = Ferrite.Dirichlet(
            :u,
            getnodeset(rve.grid, "cornerset"),
            (x, t) -> 0.0,#zero(Vec{dim,Float64}),
            [dim]#(dim-1)
        )
        add!(rve.ch, dbc)

    elseif rve.BC_TYPE == STRONG_PERIODIC()
        nodedofs = extract_nodedofs(rve.dh, :u)
        Γ_rightnodes = faceset_to_nodeset(rve.grid, getfaceset(rve.grid, "Γ⁺" ))
        Γ_leftnodes = faceset_to_nodeset(rve.grid, getfaceset(rve.grid, "Γ⁻") )
        facepairs = ["right"=>"left"]
        @info "Adding linear constraints"
        if dim ==3
            push!(facepairs, "back"=>"front")
        end
        nodepairs, masternodes = search_nodepairs(rve.grid, facepairs, rve.L◫)
        add_linear_constraints!(rve.grid, rve.ch, nodedofs, macroscale, nodepairs, masternodes)
    elseif rve.BC_TYPE isa STRONG_PERIODIC_WITH_PAIRS
        nodepairs = rve.BC_TYPE.nodepairs
        masternodes = first(nodepairs)[2]
        add_linear_constraints!(rve.grid, rve.ch, nodedofs, macroscale, nodepairs, masternodes)
    end
    @info "Closing ch"
    close!(rve.ch)
    update!(rve.ch, 0.0)
    
    @info "creating sparsity patters"
    rve.matrices.Kuu = create_sparsity_pattern(rve.dh, rve.ch)

end

function _assemble_volume!(rve::RVE, macroscale::MacroParameters, state::State)

    fill!(rve.matrices.Kuu, 0.0)
    fill!(rve.matrices.Kλu, 0.0)
    fill!(rve.matrices.fuu, 0.0)

    for (partid, part) in enumerate(rve.parts)
        material = part.material
        cellset = part.cellset
        matstates = state.partstates[partid].materialstates

        _assemble!(rve, macroscale, material, matstates, cellset, state.a)
    end

end


function _assemble!(rve::RVE{dim}, macroscale::MacroParameters, material::AbstractMaterial, materialstates::Vector{Vector{MS}}, cellset::Vector{Int}, a::Vector{Float64}) where {dim, MS<:AbstractMaterialState}

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
        #=for d in 1:dim
            fill!(fλ, 0.0)
            diffresult_λ = integrate_fλu!(diffresult_λ, fλ, cv_u, ae, d); 
            Ke = DiffResults.jacobian(diffresult_λ)
            matrices.Kλu[[d], udofs] += -Ke * (1/Ω◫)
        end =#

        for d in 1:dim-1
            fill!(fλ, 0.0)
            diffresult_λ = integrate_fλθ!(diffresult_λ, fλ, cv_u, X, ae, d); 
            Ke = DiffResults.jacobian(diffresult_λ)
            matrices.Kλu[[d], udofs] += Ke * (1/I◫)
        end
    end

    for d in 1:dim-1
        matrices.fext_λ[d] +=  -macroscale.θ[d]
    end

end

function assemble_face!(rve::RVE{dim}, macroscale::MacroParameters, a::Vector{Float64}) where dim

    (; udofs, X, ae, fμ, diffresult_μ) = rve.cache
    (; grid, dh, fv_u, ip_μ)    = rve
    (; matrices) = rve
    (; diffresult_μ) = rve.cache
    (; A◫) = rve

    μdofs = collect(1:rve.nμdofs)

    fill!(rve.matrices.Kμu, 0.0)
    fill!(rve.matrices.fext_μ, 0.0)

    Γ⁺ = union( getfaceset(grid, "Γ⁺") ) |> collect
    Γ⁻ = union( getfaceset(grid, "Γ⁻") ) |> collect
    
    #a = zeros(Float64, ndofs(dh)) # TODO: where to put this?

    @info "Gamma +"
    for (cellid, fidx) in Γ⁺
        celldofs!(udofs, dh, cellid)
        getcoordinates!(X, dh.grid, cellid)
        disassemble!(ae, a, udofs)

        reinit!(fv_u, X, fidx)

        fill!(fμ, 0.0)
        diffresult_μ = integrate_fμu!(diffresult_μ, fμ, fv_u, ip_μ, X, ae);

        #fe = DiffResults.value(diffresult_μ)
        Ke = DiffResults.jacobian(diffresult_μ)
        matrices.Kμu[μdofs, udofs] += -Ke  * (1/A◫) 

        fill!(fμ, 0.0)
        integrate_fμ_ext!(fμ, fv_u, ip_μ, X, macroscale.∇u, macroscale.∇w, macroscale.∇θ);
        matrices.fext_μ[μdofs] += fμ * (1/A◫)
    end

    @info "Gamma -"
    for (cellid, fidx) in Γ⁻
        
        celldofs!(udofs, dh, cellid)
        getcoordinates!(X, dh.grid, cellid)        
        disassemble!(ae, a, udofs)
        
        reinit!(fv_u, X, fidx)
        
        fill!(fμ, 0.0)
        diffresult_μ = integrate_fμu!(diffresult_μ, fμ, fv_u, ip_μ, X, ae);

        #fe = DiffResults.value(diffresult_μ)
        Ke = DiffResults.jacobian(diffresult_μ)
        matrices.Kμu[μdofs, udofs] += Ke  * (1/A◫) 

        fill!(fμ, 0.0)
        integrate_fμ_ext!(fμ, fv_u, ip_μ, X, macroscale.∇u, macroscale.∇w, macroscale.∇θ);
        matrices.fext_μ[μdofs] += -fμ * (1/A◫)
    end

end

function solve_it!(rve::RVE, state::State)
    if rve.SOLVE_STYLE == SOLVE_FULL
        _solve_it_full!(rve::RVE, state::State)
    elseif rve.SOLVE_STYLE == SOLVE_SCHUR
        _solve_it_schur!(rve::RVE, state::State)
    end
    #if rve.BC_TYPE == STRONG_PERIODIC || rve.BC_TYPE == DIRICHLET
    #    _solve_it_strong_periodic(rve, state)
    #elseif rve.BC_TYPE == WEAK_PERIODIC
    #    _solve_it_weak_periodic(rve, state)
    #else
    #    error("WRONG BC_TYPE")
    #end
end

function _solve_it_schur!(rve::RVE, state::State)

    (; dh, ch,  matrices)   = rve
    (; nμdofs, nλdofs) = rve

    nλdofs = rve.nλdofs 
    nμdofs = rve.nμdofs 
    nudofs = ndofs(ch.dh)
    nλμdofs = nλdofs + nμdofs

    K = matrices.Kuu
    RHS = zeros(Float64, nudofs, 1 + nλdofs + nμdofs)
    fext_u = zeros(Float64, nudofs)
    #Ferrite._condense_sparsity_pattern!(K, ch.acs)

    fμλ = vcat(matrices.fext_μ, matrices.fext_λ)
    C = vcat( matrices.Kμu, matrices.Kλu )
    Ct = copy(C')
    b = zeros(Float64, ndofs(ch.dh)); 
    b[ch.prescribed_dofs] .= ch.inhomogeneities

    apply!(K, fext_u, ch)
    condense_rhs!(Ct, ch) 
    
    RHS[:, 1] .= fext_u
    RHS[:, (1:nλμdofs) .+ 1] .= -Ct

    @time LHS = K\RHS
    ub = LHS[:,1]               # ub =  K\fext_u
    Uλ = LHS[:,(1:(nλμdofs)) .+ 1] # Uλ = -K\Ct 

    ub[Ferrite.prescribed_dofs(ch)] .= 0.0 #apply_zero!(ub, ch)

    μλ = (Ct'*Uλ)\(-C*b - Ct'*ub + fμλ)

    state.a[(1:nudofs)           ] .= ub + Uλ*μλ
    state.a[(1:nλμdofs) .+ nudofs] .= μλ

    apply!(state.a, ch)
end

function _solve_it_full!(rve::RVE, state::State)

    (; dh, ch, matrices)   = rve
    (; nμdofs, nλdofs) = rve

    fext_u = zeros(Float64, ndofs(ch.dh))

    KK = vcat(hcat(matrices.Kuu, matrices.Kμu', matrices.Kλu'),
              hcat(matrices.Kμu, zeros(Float64, nμdofs, nμdofs), zeros(Float64,nμdofs,nλdofs)),
              hcat(matrices.Kλu, zeros(Float64, nλdofs, nμdofs), zeros(Float64,nλdofs,nλdofs)))

    ff = vcat(zeros(Float64, ndofs(dh)), matrices.fext_μ, matrices.fext_λ)
    
    Ferrite._condense_sparsity_pattern!(KK, ch.acs)

    apply!(KK, ff, ch)
    time1 = @elapsed(state.a .= KK\ff)
    apply!(state.a, ch)
    @info "Solving the full sytem took $time1 seconds"
end

function _solve_it_weak_periodic(rve::RVE, state::State)

    (; dh, ch, matrices)   = rve
    (; nμdofs, nλdofs) = rve

    KK = vcat(hcat(matrices.Kuu, matrices.Kμu', matrices.Kλu'),
        hcat(matrices.Kμu, zeros(Float64, nμdofs, nμdofs), zeros(Float64,nμdofs,nλdofs)),
        hcat(matrices.Kλu, zeros(Float64, nλdofs, nμdofs), zeros(Float64,nλdofs,nλdofs)))

    ff = vcat(zeros(Float64, ndofs(dh)), matrices.fext_μ, matrices.fext_λ)
        
        
    aa .= KK\ff

    nλdofs = rve.nλdofs 
    nudofs = ndofs(ch.dh)

    K = matrices.Kuu
    RHS = zeros(Float64, nudofs, 1 + nλdofs)
    fext_u = zeros(Float64, nudofs)
    #Ferrite._condense_sparsity_pattern!(K, ch.acs)

    fλ = matrices.fext_λ
    C = copy(matrices.Kλu)
    Ct = copy(matrices.Kλu)'
    b = zeros(Float64, ndofs(ch.dh)); 
    b[ch.prescribed_dofs] .= ch.inhomogeneities

    apply!(K, fext_u, ch)
    condense_rhs!(Ct, ch) 
    
    RHS[:, 1] .= fext_u
    RHS[:, (1:nλdofs) .+ 1] .= -Ct

    @time LHS = K\RHS
    ub = LHS[:,1]               # ub =  K\fext_u
    Uλ = LHS[:,(1:nλdofs) .+ 1] # Uλ = -K\Ct 

    ub[Ferrite.prescribed_dofs(ch)] .= 0.0 #apply_zero!(ub, ch)

    λ = (Ct'*Uλ)\(-C*b - Ct'*ub + fλ)

    state.a[(1:nudofs)          ] .= ub + Uλ*λ
    state.a[(1:nλdofs) .+ nudofs] .= λ

    apply!(state.a, ch)
    
    #μdofs = (1:nμdofs) .+ ndofs(dh)
    #λdofs = (1:nλdofs) .+ ndofs(dh) .+ nμdofs

    #μ = a[μdofs]
    #λ = a[λdofs]
end


function _solve_it_strong_periodic(rve::RVE, state::State)

    (; dh, ch, matrices)   = rve
    (; nμdofs, nλdofs) = rve

    #=
    fext_u = zeros(Float64, ndofs(ch.dh))

    K = vcat(hcat(matrices.Kuu, matrices.Kλu'),
             hcat(matrices.Kλu, zeros(Float64, nλdofs, nλdofs)))
    f = vcat(fext_u, matrices.fext_λ)
    
    Ferrite._condense_sparsity_pattern!(K, ch.acs)

    @show size(matrices.Kλu)
    apply!(K, f, ch)
    state.a .= K\f
    @show state.a[end-1:end]
    apply!(state.a, ch)
    @show state.a[end-1:end]=#
    #state.a[end - 1:end] = [-0.1324798100769443, 4.184280901039025e-18]
    #state.a[end - 1:end] = [-0.1324798100769443, 4.184280901039025e-18]
    #-0.06743940990516334 -0.038461538461538464
   # state.a[end - 4:end] = [0.0023190336868656783, -3.2790763838489505e-5, -0.0049601617569132225, -0.1528202442617568, -0.0011312339754665324]
    
   
    nλdofs = rve.nλdofs 
    nudofs = ndofs(ch.dh)

    K = matrices.Kuu
    RHS = zeros(Float64, nudofs, 1 + nλdofs)
    fext_u = zeros(Float64, nudofs)
    #Ferrite._condense_sparsity_pattern!(K, ch.acs)

    fλ = matrices.fext_λ
    C = copy(matrices.Kλu)
    Ct = copy(matrices.Kλu)'
    b = zeros(Float64, ndofs(ch.dh)); 
    b[ch.prescribed_dofs] .= ch.inhomogeneities
#=
    apply!(K, fext_u, ch)
    condense_rhs!(Ct, ch) 
    
    RHS[:, 1] .= fext_u
    RHS[:, (1:nλdofs) .+ 1] .= -Ct

    @time LHS = K\RHS
    ub = LHS[:,1]               # ub =  K\fext_u
    Uλ = LHS[:,(1:nλdofs) .+ 1] # Uλ = -K\Ct 

    ub[Ferrite.prescribed_dofs(ch)] .= 0.0 #apply_zero!(ub, ch)

    λ = (Ct'*Uλ)\(-C*b - Ct'*ub + fλ)

    state.a[(1:nudofs)          ] .= ub + Uλ*λ
    state.a[(1:nλdofs) .+ nudofs] .= λ

    apply!(state.a, ch)=#

    
    T, b = Ferrite.create_constraint_matrix(ch)
    ûb = (T'*K*T)\(-T'*K*b)
    Uλ = (T'*K*T)\(-T'*C')
    λ = (C*T*Uλ)\(-C*b - C*T*ûb + fλ)
    apply!(state.a, ch) 
    
    #@show norm(state.a)

    @show λ
    #error("sdf")
end

function condense_rhs!(f::AbstractVecOrMat, ch::ConstraintHandler)
    
    acs = ch.acs
    ndofs = size(f, 1)
    distribute = Dict{Int,Int}(acs[c].constrained_dof => c for c in 1:length(acs))

    for col in 1:ndofs
        dcol = get(distribute, col, 0)
        if dcol != 0
            ac = acs[dcol]
            for (d,v) in ac.entries
                f[d,:] += f[col,:] * v
            end
            @assert ac.constrained_dof == col
            f[ac.constrained_dof,:] .= 0.0
        end
    end
end

function search_nodepairs(grid::Grid{dim,C,T}, facepairs::Vector{<:Pair}, side_lengths::NTuple{3,T}) where {dim,C,T}

    SEARCH_TOL = 1e-3
    nodepairs = Dict{Int,Int}()
    Γ⁺_nodes = Int[]
    Γ⁻_nodes = Int[]

    dir = 0
    masternodes = []
    for (Γ⁺_name, Γ⁻_name) in facepairs
        
        dir += 1
        Γ⁺_nodes = collect(getnodeset(grid, Γ⁺_name))
        Γ⁻_nodes = collect(getnodeset(grid, Γ⁻_name))

        for nodeid_r in Γ⁺_nodes

            Xr = grid.nodes[nodeid_r].x

            Xoffset = Vec{dim,T}(i -> i==dir ? side_lengths[dir] : 0.0)
            found_pair = false
            mindist = Inf

            for nodeid_l in Γ⁻_nodes

                Xl = grid.nodes[nodeid_l].x

                dist = norm(Xr - (Xl + Xoffset))
                mindist = min(mindist, dist) #For debugging
                if isapprox(dist, 0.0, atol = SEARCH_TOL) #Vec(-4.440892098500626e-16, 0.0) ≈ zero(Vec{2}) ?????
                    
                    found_pair = true
                    
                    if haskey(nodepairs, nodeid_r) #Masternoded har constraint: alltså det är en hörnnod
                        nodeid_r′ = nodepairs[nodeid_r]
                        #@info "$nodeid_l => $nodeid_r, but $nodeid_r => $nodeid_r′, remapping $nodeid_l => $nodeid_r′."
                        nodepairs[nodeid_l] = nodeid_r′
                        push!(masternodes, nodeid_r′)
                    elseif haskey(nodepairs, nodeid_l) && nodepairs[nodeid_l] == nodeid_r
                       # @info "$nodeid_l => $nodeid_r already in the set, skipping."
                    elseif haskey(nodepairs, nodeid_l)
                        #@info "$nodeid_l => $nodeid_r, but $nodeid_l => $(nodepairs[nodeid_l]) already, skipping."
                    else
                        #@info "$nodeid_l => $nodeid_r "
                        push!(nodepairs, nodeid_l => nodeid_r)
                    end
                    
                    break;
                end

            end

            if found_pair == false
                error("No pair node found, $nodeid_r, mindist: $mindist")
            end

        end
    end

    return nodepairs, masternodes

    #check
    #=@info "check that no master nodes are slave nodes"
    for (k,v) in nodepairs
        if haskey(nodepairs, v)
            @info "masternode $v exisits as slave"
        end
        #@info "$k => $v"
    end

    @info "check that no Γ- nodes has constraints"
    for (Γ⁺_name, Γ⁻_name) in facepairs
        Γ⁻_nodes = collect(getnodeset(grid, Γ⁻_name))
        for n in Γ⁻_nodes
            if !haskey(nodepairs, n)
                @info "$n does not have a constraint"
            end
        end
    end=#

end

function add_linear_constraints!(grid::Grid{dim}, ch::ConstraintHandler, nodedofs::Matrix{Int}, macroscale::MacroParameters, nodepairs#=::Dict{Int,Int}=#, masternodes) where dim

    for (nodeid_s, nodeid_m) in nodepairs
        
        xm = grid.nodes[nodeid_m].x
        xs = grid.nodes[nodeid_s].x
        
        x_jump = xs-xm
        
        ∇u = increase_dim(macroscale.∇u)
        ∇θ = increase_dim(macroscale.∇θ)
        ∇w = increase_dim(macroscale.∇w)

        e₃ = basevec(Vec{dim,Float64})[dim]

        z = xm[dim]

        b = ∇u⋅x_jump - z*∇θ⋅x_jump + ∇w⋅x_jump * e₃

        dof_s = nodedofs[:,nodeid_s]
        dof_l = nodedofs[:,nodeid_m]
        
        for d in 1:dim #ncomponents?
            lc = AffineConstraint(dof_s[d], [ (dof_l[d] => 1.0) ], b[d])
            add!(ch, lc)
        end
        #add!(ch, Dirichlet(:u, Set([nodeid_r]), (x,t)->b[dim], [dim]))

    end

    #Lock a master node
    lock_masternode = -1
    if length(masternodes) == 0
        @assert dim == 2
        lock_masternode = first(nodepairs)[2] #Pick the first dof on Γ+
    else
        lock_masternode = first(masternodes)
    end
    
    add!(ch, Ferrite.Dirichlet(:u, Set([lock_masternode]), (x,t) -> zero(Vec{dim}), 1:dim))

end

function extract_nodedofs(dh::DofHandler, field::Symbol)

    
    fieldidx = Ferrite.find_field(dh, field)
    field_dim = Ferrite.getfielddim(dh, fieldidx)
    offset = Ferrite.field_offset(dh, :u)
    _celldofs = fill(0, ndofs_per_cell(dh))
    ncomps = field_dim
    
    nnodes = getnnodes(dh.grid)
    node_dofs = zeros(Int, ncomps, nnodes)
    visited = falses(nnodes)

    for cellid in 1:getncells(dh.grid)

        cell = dh.grid.cells[cellid]

        celldofs!(_celldofs, dh, cellid) # update the dofs for this cell
        for idx in 1:length(cell.nodes)
            node = cell.nodes[idx]
            if !visited[node]
                noderange = (offset + (idx-1)*field_dim + 1):(offset + idx*field_dim) # the dofs in this node
                for (i,c) in enumerate(1:ncomps)
                    node_dofs[i,node] = _celldofs[noderange[c]]
                end
                visited[node] = true
            end
        end
        
    end

    return node_dofs
end

function faceset_to_nodeset(grid::Grid, set::Set{FaceIndex})
    nodeset = Set{Int}()
    for (cellidx, faceidx) in set   
        facenodes = Ferrite.faces(grid.cells[cellidx])[faceidx]
        for node in facenodes
            push!(nodeset, node)
        end
    end
    return nodeset
end

include("response.jl")

export MacroParameters
export RVESubPartState
export State
end

