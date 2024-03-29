
module MultiScale

using MaterialModels
using Ferrite
#using FerriteGmsh
using LinearAlgebra
using Tensors
using ForwardDiff
using DiffResults: DiffResult
using SparseArrays
using Random, Distributions
using Plots: plot, plot!, scatter3d!, plot3d!, surface!
using IterativeSolvers
using SymRCM
using LinearSolve: solve, LinearProblem, LinearSolve
using BlockArrays
using MKLSparse # Snabbare om man kör men julia --threads 4

using Preconditioners
using IncompleteLU
using AlgebraicMultigrid
function IterativeSolvers.Identity(a)
    return IterativeSolvers.Identity()
end

export WEAK_PERIODIC, STRONG_PERIODIC, STRONG_PERIODIC_FERRITE, DIRICHLET, RELAXED_DIRICHLET, RELAXED_PERIODIC

include("integrals.jl")
include("extra_materials.jl")
include("sampledomain.jl")
include("utils.jl")
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
    dir1 = ( isapprox(abs(n[1]), 1.0, atol=1e-2) && 
             isapprox(abs(n[2]), 0.0, atol=1e-2) && 
             isapprox(abs(n[3]), 0.0, atol=1e-2)) ? true : false
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
    dir1 = ( isapprox(abs(n[1]), 1.0, atol=1e-2) && 
             isapprox(abs(n[2]), 0.0, atol=1e-2) && 
             isapprox(abs(n[3]), 0.0, atol=1e-2)) ? true : false

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
    Kμu::SparseMatrixCSC{Float64,Int}
    Kλu::SparseMatrixCSC{Float64,Int}

    fuu::Vector{Float64}
    fμu::Vector{Float64}
    fλu::Vector{Float64}

    fext_u::Vector{Float64}
    uM    ::Vector{Float64}
    fext_μ::Vector{Float64}
    fext_λ::Vector{Float64}

    check_z::Vector{Float64} #Check value ∫ z dΩ
    check_x::Vector{Float64}
end

function Matrices(dh::DofHandler, ch::ConstraintHandler, nμ::Int, nλ::Int)
    Kuu = spzeros(Float64, ndofs(dh), ndofs(dh)) #create_sparsity_pattern(dh, ch)
    return Matrices(
        Kuu, 
        spzeros(nμ, ndofs(dh)), 
        spzeros(nλ, ndofs(dh)), 
        zeros(Float64, ndofs(dh)), 
        zeros(Float64, ndofs(dh)), 
        zeros(Float64, ndofs(dh)), 
        zeros(Float64, ndofs(dh)),
        zeros(Float64, ndofs(dh)),
        zeros(Float64, nμ), 
        zeros(Float64, nλ), 
        [0.0], 
        [0.0]
    )
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

struct RVECache{dim, D <: DiffResult}
    udofs::Vector{Int}
    X::Vector{Vec{dim,Float64}}
    ae::Vector{Float64}
    fu::Vector{Float64}
    ke::Matrix{Float64}
    fμ::Vector{Float64}
    ke_μ::Matrix{Float64}
    fλ::Vector{Float64}
    ke_λ::Matrix{Float64}
    diffresult_u::D#DiffResult
    diffresult_μ::D#DiffResult
    diffresult_λ::D#DiffResult
end

function RVECache(dim::Int, nnodes, nudofs_per_cell, nμdofs, nλdofs)

    udofs = zeros(Int, nudofs_per_cell)
    X = zeros(Vec{dim,Float64}, nnodes)
    ae = zeros(Float64, nudofs_per_cell)

    ae = zeros(Float64, nudofs_per_cell)
    fu = zeros(Float64, nudofs_per_cell)
    ke = zeros(Float64, nudofs_per_cell, nudofs_per_cell)
    fμ = zeros(Float64, nμdofs)
    ke_μ = zeros(Float64, nμdofs, nudofs_per_cell)
    fλ = zeros(Float64, 1)
    ke_λ = zeros(Float64, 1, nudofs_per_cell)

    diffresult_u = DiffResults.JacobianResult(fu, ae)
    diffresult_μ = DiffResults.JacobianResult(fμ, ae)
    diffresult_λ = DiffResults.JacobianResult(fλ, ae)

    return RVECache{dim,typeof(diffresult_u)}(udofs, X, ae, fu ,ke, fμ, ke_μ, fλ, ke_λ, diffresult_u, diffresult_μ, diffresult_λ)
end

abstract type BCType end
struct WEAK_PERIODIC <: BCType end
struct STRONG_PERIODIC  <: BCType end
struct STRONG_PERIODIC_FERRITE  <: BCType end
struct RELAXED_DIRICHLET  <: BCType end
struct RELAXED_PERIODIC  <: BCType end
struct DIRICHLET <: BCType end
@enum SolveStyle SOLVE_SCHUR SOLVE_FULL

struct RVE{dim, CACHE<:RVECache, LS, PR}

    grid::Grid{dim}
    dh::DofHandler{dim}
    ch::ConstraintHandler 
    parts::Vector{RVESubPart}

    cache::CACHE#RVECache{3,D}

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
    nudofs::Int # Total number of u-dofs
    nμdofs::Int
    nλdofs::Int

    #
    BC_TYPE::BCType
    SOLVE_STYLE::SolveStyle
    linearsolver::LS
    preconditioner::PR
    VOLUME_CONSTRAINT::Bool    
    EXTRA_PROLONGATION::Bool
    PERFORM_CHECKS::Bool
    SOLVE_FOR_FLUCT::Bool
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

function rvecenter(grid::Grid{dim}) where dim
    
    maxx = zeros(Float64, dim)
    minx = zeros(Float64, dim)

    for node in grid.nodes
        for d in 1:dim
            maxx[d] = max(maxx[d], node.x[d])
            minx[d] = min(minx[d], node.x[d])
        end
    end
    return Vec{dim}( d-> (maxx[d] + minx[d])/2 )
end


function RVE(; 
    grid::Grid{dim}, 
    parts::Vector{RVESubPart},
    BC_TYPE::BCType, 
    SOLVE_STYLE::SolveStyle = SOLVE_FULL,
    ip_u::Interpolation = Ferrite.default_interpolation(getcelltype(grid)),
    VOLUME_CONSTRAINT = true,
    EXTRA_PROLONGATION = false,
    PERFORM_CHECKS = false,
    SOLVE_FOR_FLUCT = false,
    LINEAR_SOLVER = nothing,
    PRECON  = IterativeSolvers.Identity,) where dim

    #Get size of rve
    x_center = rvecenter(grid)
    if !isapprox(norm(x_center), 0.0, atol = 1e-14)
        error("RVE must be centered $(x_center)")
    end
    side_length = ntuple( d -> rvesize(grid; dir = d), dim)

    #TODO: Automatically add face and node sets
    #Check if all cells are included in a cellset
    _check_cellsets(parts, getncells(grid))

    #Dofhandler
    dh = DofHandler(grid)
    push!(dh, :u, dim, ip_u)
    close!(dh)

    #ConstraintHandler
    ch = ConstraintHandler(dh)
    #close!(ch)

    celltype = getcelltype(grid)
    ip_geo = Ferrite.default_interpolation(celltype)
    refshape = Ferrite.getrefshape(ip_u)
    @info "$refshape, $celltype, $ip_u"

    #Element
    qr        = Ferrite._mass_qr(ip_u)
    qr_face   = Ferrite._mass_qr( Ferrite.getlowerdim(ip_u) )

    cv_u = CellVectorValues(qr, ip_u, ip_geo)
    fv_u = FaceVectorValues(qr_face, ip_u, ip_geo)

    #
    nnodes = getnbasefunctions(ip_geo)
    nudofs = ndofs(dh)
    nudofs_per_cell = ndofs_per_cell(dh)
    ip_μ = nothing
    nμdofs = 0

    @assert( haskey(grid.facesets, "right") )
    @assert( haskey(grid.facesets, "left") )
    if dim == 3
        @assert( haskey(grid.facesets, "back") )
        @assert( haskey(grid.facesets, "front") )
    end

    if BC_TYPE == WEAK_PERIODIC()
        ip_μ   = TractionInterpolation{dim}()
        nμdofs = getnbasefunctions(ip_μ)
        @assert( SOLVE_STYLE == SOLVE_FULL)
    elseif BC_TYPE == DIRICHLET()
    elseif BC_TYPE == STRONG_PERIODIC() 
    elseif BC_TYPE == STRONG_PERIODIC_FERRITE() 
    elseif BC_TYPE == RELAXED_DIRICHLET()
        ip_μ = RelaxedDirichletInterpolation{dim}()
        nμdofs = getnbasefunctions(ip_μ)  
    elseif BC_TYPE == RELAXED_PERIODIC()
        ip_μ = RelaxedDirichletInterpolation{dim}()
        nμdofs = getnbasefunctions(ip_μ)  
    else
        error("Wrong BCTYPE")
    end
    
    nλdofs = 0
    if VOLUME_CONSTRAINT
        nλdofs += dim-1
        #nλdofs += dim
    end
    
    cache = RVECache(dim, nnodes, nudofs_per_cell, nμdofs, nλdofs)
    matrices = Matrices(dh, ch, nμdofs, nλdofs)

    #
    h = side_length[dim]
    Ω◫ = prod(side_length)
    A◫ = Ω◫ / h
    I◫ = A◫*h^3/12

    #TODO: Automatically build linear solver and precon
    return RVE(grid, dh, ch, parts, cache, matrices, cv_u, fv_u, ip_μ, side_length, Ω◫, A◫, I◫, nudofs, nμdofs, nλdofs, BC_TYPE, SOLVE_STYLE, LINEAR_SOLVER, PRECON, VOLUME_CONSTRAINT, EXTRA_PROLONGATION, PERFORM_CHECKS, SOLVE_FOR_FLUCT)
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
    @assert getnquadpoints( rve.cv_u ) == length(  first(first(partstates).materialstates) )
    _ndofs = ndofs(rve.dh) + rve.nμdofs + rve.nλdofs
    return State(zeros(Float64, _ndofs), partstates)
end

function solve_rve(rve::RVE{dim}, macroscale::MacroParameters, state::State) where dim
    @info "applying macroscale"
    _apply_macroscale!(rve, macroscale, state)

    @info "assembling volume"
    @time _assemble_volume!(rve, macroscale, state)

    @info "Evaluating uM"
    evaluate_uᴹ!(rve.matrices.uM, rve, macroscale)
    
    @info "Solving it"
    time = @elapsed(solve_it!(rve, state))


    return time
end

function prolongation(x::Vec{dim,T}, x̄::Vec{dim,Float64}, macroparamters::MacroParameters, with_extra) where {dim,T}

    #return prolongation2(x, x̄, macroparamters, with_extra)
    
    (; ∇u, ∇w, ∇θ, u, w, θ) = macroparamters
    
    d = dim-1

    e = basevec(Vec{dim,Float64})
    z = x[dim]

    φu(α) = e[α]
    φh(α,β) = ((x - x̄) ⋅ e[β])*e[α]

    φw() = e[dim]
    φg(α) = ((x - x̄) ⋅ e[α])*e[3]

    φθ(α) = -z*e[α]
    
    function φκ(α,β) 
        _φ = zero(Vec{dim,T})
        _φ = -z*(x - x̄)⋅(e[β]⊗e[α]) 

        if with_extra 
            _φ += ( 0.5*(e[α]⊗e[β])⊡((x-x̄)⊗(x-x̄)) )*e[3]
        end

        return _φ
    end

    um = zero(Vec{dim,T})

    #
    for α in 1:d
        um += φu(α) * u[α]
    end

    for α in 1:d, β in 1:d
        um += φh(α,β) * ∇u[α,β]
    end

    #
    um += w*φw()
    
    for α in 1:d
        um += φg(α) * ∇w[α]
    end

    #
    for α in 1:d
        um += φθ(α) * θ[α]
    end

    for α in 1:d, β in 1:d
        um += φκ(α,β) * ∇θ[α,β]
    end

    return um
   
end


function prolongation2(x::Vec{dim,T}, x̄::Vec{dim,Float64}, macroparamters::MacroParameters, with_extra) where {dim,T}

    (; ∇u, ∇w, ∇θ, u, w, θ) = macroparamters

    ∇u = increase_dim(∇u)
    ∇w = increase_dim(∇w)
    ∇θ = increase_dim(∇θ) 
    u  = increase_dim(u)
    #w  = increase_dim(w)
    θ  = increase_dim(θ)

    e = basevec(Vec{dim,Float64})
    z = x[dim]
    
    Î = (e[1]⊗e[1]) + (e[2]⊗e[2])
    Îz = (e[3]⊗e[3])

    um = zero(Vec{dim,T})
    um += u
    um += w*e[3]
    um += -z*θ
    um += ∇u ⋅ (x -x̄) 
    um += (∇w ⋅ (x -x̄))*e[3]
    um += -z*∇θ ⋅ (x -x̄)
    if with_extra
        #@show Î⊡((x-x̄)⊗(x-x̄))
        um += 0.5*((∇θ⊡((x-x̄)⊗(x-x̄)))*e[3])
    end

    return um

end

function _apply_macroscale!(rve::RVE{dim}, macroscale::MacroParameters, state::State) where dim

    ∇u = increase_dim(macroscale.∇u)
    ∇θ = increase_dim(macroscale.∇θ)
    ∇w = increase_dim(macroscale.∇w)

    e₃ = basevec(Vec{dim}, dim)
    x̄ = zero(Vec{dim})

    uᴹ = prolongation
    solve_uˢ = rve.SOLVE_FOR_FLUCT

    reset_constrainthandler!(rve.ch)

    #TODO: Do this nicer:
    Γ⁺ = union( getfaceset(rve.grid, "right") )
    Γ⁻ = union( getfaceset(rve.grid, "left") )
    if dim == 3
        union!(Γ⁺, getfaceset(rve.grid, "back"))
        union!(Γ⁻, getfaceset(rve.grid, "front"))
    end

    @info "Adding $(typeof(rve.BC_TYPE))"

    if rve.BC_TYPE == WEAK_PERIODIC()
        assemble_face!(rve, macroscale, state.a)

        dbc = Ferrite.Dirichlet(
            :u,
            getnodeset(rve.grid, "cornerset"),
            (x, t) -> solve_uˢ ? zero(Vec{dim}) : [macroscale.u..., macroscale.w],
            1:dim
        )
        add!(rve.ch, dbc)

    elseif rve.BC_TYPE == DIRICHLET()

        dbc = Ferrite.Dirichlet(
            :u,
            union(
                Γ⁺,
                Γ⁻
            ),
            (x, t) -> solve_uˢ ? zero(Vec{dim}) : uᴹ(x, x̄, macroscale, rve.EXTRA_PROLONGATION),
            1:dim
        )
        add!(rve.ch, dbc)

    elseif rve.BC_TYPE == RELAXED_DIRICHLET()
        assemble_face!(rve, macroscale, state.a)

        dbc = Ferrite.Dirichlet(
            :u,
            union(
                Γ⁺,
                Γ⁻
            ),
            (x, t) -> solve_uˢ ? zero(Vec{dim-1}) : uᴹ(x, x̄, macroscale, rve.EXTRA_PROLONGATION)[1:dim-1],
            1:(dim-1)
        )
        add!(rve.ch, dbc)

        @info "Locking corner node"
        dbc = Ferrite.Dirichlet(
            :u,
            getnodeset(rve.grid, "cornerset"),
            (x, t) -> solve_uˢ ? 0.0 : uᴹ(x, x̄, macroscale, rve.EXTRA_PROLONGATION)[dim],
            [dim]
        )
        add!(rve.ch, dbc)
    elseif rve.BC_TYPE == STRONG_PERIODIC()
        nodedofs = extract_nodedofs(rve.dh, :u)
        addnodeset!(rve.grid, "right", faceset_to_nodeset(rve.grid, getfaceset(rve.grid, "right")))
        addnodeset!(rve.grid, "left", faceset_to_nodeset(rve.grid, getfaceset(rve.grid, "left")))
        facepairs = ["right"=>"left"]
        if dim ==3
            push!(facepairs, "back"=>"front")
            addnodeset!(rve.grid, "back", faceset_to_nodeset(rve.grid, getfaceset(rve.grid, "back")))
            addnodeset!(rve.grid, "front", faceset_to_nodeset(rve.grid, getfaceset(rve.grid, "front")))
        end
        nodepairs, masternodes = search_nodepairs(rve.grid, facepairs, rve.L◫)
        masternode = getnodeset(rve.grid, "cornerset") |> first
        #masternode = masternodes[4]
        
        @info "Adding linear constraints"
        add_linear_constraints!(rve.grid, rve.ch, nodedofs, macroscale, nodepairs, masternode, rve.EXTRA_PROLONGATION, rve.SOLVE_FOR_FLUCT)

        #Delete the sets from the grid
        delete!(rve.grid.nodesets, "right")
        delete!(rve.grid.nodesets, "left")
        if dim ==3
            delete!(rve.grid.nodesets, "front")
            delete!(rve.grid.nodesets, "back")
        end
    elseif rve.BC_TYPE == STRONG_PERIODIC_FERRITE()

        #face_map = collect_periodic_faces(rve.grid, "Γ⁺", "Γ⁻", x -> rotate(x, basevec(Vec{dim}, dim), 1π))

        face_map = Ferrite.PeriodicFacePair[]
        collect_periodic_faces!(face_map, rve.grid, "right", "left", x -> x+Vec((-rve.L◫[1], 0.0, 0.0)))
        collect_periodic_faces!(face_map, rve.grid, "back", "front", x -> x+Vec((0.0, -rve.L◫[2], 0.0)))
        @assert length(face_map) == (length(getfaceset(rve.grid, "right"))+length(getfaceset(rve.grid, "back")))

        pbc = Ferrite.PeriodicDirichlet(
            :u,
            face_map,
            (x, t) -> solve_uˢ ? zero(Vec{dim}) : uᴹ(x, x̄, macroscale, rve.EXTRA_PROLONGATION),
            1:dim
        )
        add!(rve.ch, pbc)

        #=dbc = Ferrite.Dirichlet(
            :u,
            getnodeset(rve.grid, "cornerset"),
            (x, t) -> solve_uˢ ? zero(Vec{dim}) : uᴹ(x, x̄, macroscale, rve.EXTRA_PROLONGATION),
            1:dim
        )
        add!(rve.ch, dbc)=#
    elseif rve.BC_TYPE == RELAXED_PERIODIC()
        assemble_face!(rve, macroscale, state.a)

        face_map = Ferrite.PeriodicFacePair[]
        collect_periodic_faces!(face_map, rve.grid, "right", "left", x -> x+Vec((-rve.L◫[1], 0.0, 0.0)))
        collect_periodic_faces!(face_map, rve.grid, "back", "front", x -> x+Vec((0.0, -rve.L◫[2], 0.0)))
        @assert length(face_map) == (length(getfaceset(rve.grid, "right"))+length(getfaceset(rve.grid, "back")))

        pbc = Ferrite.PeriodicDirichlet(
            :u,
            face_map,
            (x, t) -> solve_uˢ ? zero(Vec{dim-1}) : uᴹ(x, x̄, macroscale, rve.EXTRA_PROLONGATION)[1:dim-1],
            1:(dim-1)
        )
        add!(rve.ch, pbc)

        dbc = Ferrite.Dirichlet(
            :u,
            getnodeset(rve.grid, "cornerset"),
            (x, t) -> solve_uˢ ? zero(Vec{dim}) : uᴹ(x, x̄, macroscale, rve.EXTRA_PROLONGATION),
            1:dim
        )
        add!(rve.ch, dbc)
    end

    if rve.VOLUME_CONSTRAINT
        for d in 1:dim-1
            rve.matrices.fext_λ[d] +=  solve_uˢ ? 0.0 : -macroscale.θ[d]
        end
        #=for d in 1:dim
            rve.matrices.fext_λ[d+2] +=  0.0#solve_uˢ ? 0.0 : -macroscale.θ[d]
        end=#
    end

    @info "Closing ch"
    close!(rve.ch)
    update!(rve.ch, 0.0)

    @info "creating sparsity patters"
    @time rve.matrices.Kuu = create_sparsity_pattern(rve.dh, rve.ch)
    @info "Size of Kuu: $(Base.summarysize(rve.matrices.Kuu)/1024^3)"

end

function _assemble_volume!(rve::RVE, macroparamters::MacroParameters, state::State)

    fill!(rve.matrices.Kuu, 0.0)
    fill!(rve.matrices.Kλu, 0.0)
    fill!(rve.matrices.fuu, 0.0)
    fill!(rve.matrices.fext_u, 0.0)
    fill!(rve.matrices.check_x, 0.0)
    fill!(rve.matrices.check_z, 0.0)


    assembler_u = start_assemble(rve.matrices.Kuu, rve.matrices.fuu, fillzero=false)
    assembler_λ = start_assemble(rve.nudofs * rve.nλdofs)

    for (partid, part) in enumerate(rve.parts)
        material = part.material
        cellset = part.cellset
        materialstates = state.partstates[partid].materialstates

        _assemble!(material, assembler_u, assembler_λ, materialstates, cellset, state.a, rve.grid, macroparamters, rve.dh, rve.cv_u, rve.cache, rve.matrices, rve.A◫, rve.Ω◫, rve.I◫, rve.VOLUME_CONSTRAINT, rve.PERFORM_CHECKS, rve.EXTRA_PROLONGATION, rve.SOLVE_FOR_FLUCT)
    end

    rve.matrices.Kλu = sparse(assembler_λ.I, assembler_λ.J, assembler_λ.V, rve.nλdofs, rve.nudofs)#end_assemble(assembler_λ)
    
    if rve.PERFORM_CHECKS
        @info "Check value ∫ z dΩ: $(rve.matrices.check_z[1])"
        @info "Check value ∫ xₚ - ̄x dΩ: $(rve.matrices.check_x[1])"
        @info ""
    end
end


function _assemble!(material::AbstractMaterial, assembler_u, assembler_λ, materialstates::Vector{Vector{MS}}, cellset::Vector{Int}, a::Vector{Float64}, grid::Grid{dim}, macroparamters, dh, cv_u, cache, matrices, A◫, Ω◫, I◫, VOLUME_CONSTRAINT, PERFORM_CHECKS, EXTRA_PROLONGATION, SOLVE_FOR_FLUCT) where {MS,dim}

    (; udofs, X, ae, fu, fλ, ke, ke_λ) = cache

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
        
        integrate_fuu2!(ke, fu, cv_u, material, mstates, ae)
        assemble!(assembler_u, udofs, 1/A◫ * ke, 1/A◫ * fu)

        #---
        #=for d in 1:dim
            fill!(fλ, 0.0)
            fill!(ke_λ, 0.0)
            integrate_fλu_2!(ke_λ, fλ, cv_u, ae, d); 
            Ferrite.assemble!(assembler_λ, Vec{1,Int}((d+2,)), udofs, ke_λ * (1/Ω◫)) #matrices.Kλu[[d], udofs] += -ke_λ * (1/Ω◫) 
        end =#

        if VOLUME_CONSTRAINT
            for d in 1:dim-1
                fill!(fλ, 0.0)
                fill!(ke_λ, 0.0)
                # AD:
                #diffresult_λ = integrate_fλθ!(diffresult_λ, fλ, cv_u, X, ae, d); 
                #Ke = DiffResults.jacobian(diffresult_λ)

                # Analytical
                integrate_fλθ_2!(ke_λ, fλ, cv_u, X, ae, d); 
                Ferrite.assemble!(assembler_λ, Vec{1,Int}((d,)), udofs, ke_λ * (1/I◫)) #matrices.Kλu[[d], udofs] += ke_λ * (1/I◫)
            end
        end

        if SOLVE_FOR_FLUCT
            fill!(fu, 0.0)
            x̄ = zero(Vec{dim})
            uᴹ(x)  = prolongation(x, x̄, macroparamters, EXTRA_PROLONGATION)
            ∇uᴹ(x) = Tensors.gradient(uᴹ, x)

            integrate_rhs!(fu, cv_u, material, mstates, X, ∇uᴹ)
            matrices.fext_u[udofs] .+= 1/A◫ * fu 
        end

        if PERFORM_CHECKS
            integrate_checks(matrices.check_z, matrices.check_x, cv_u, X)
        end
    end

end

function assemble_face!(rve::RVE{dim}, macroscale::MacroParameters, a::Vector{Float64}) where dim

    (; udofs, X, ae, fμ, ke_μ, diffresult_μ) = rve.cache
    (; grid, dh, fv_u, ip_μ)    = rve
    (; matrices) = rve
    (; diffresult_μ) = rve.cache
    (; A◫) = rve

    assembler_μ = start_assemble(rve.nudofs * rve.nμdofs)

    μdofs = collect(1:rve.nμdofs)

    fill!(rve.matrices.Kμu, 0.0)
    fill!(rve.matrices.fext_μ, 0.0)

    Γ⁺ = union( getfaceset(grid, "right") )
    Γ⁻ = union( getfaceset(grid, "left") )
    if dim == 3
        union!(Γ⁺, getfaceset(grid, "back"))
        union!(Γ⁻, getfaceset(grid, "front"))
    end
    
    @info "Gamma +"
    for (cellid, fidx) in Γ⁺
        celldofs!(udofs, dh, cellid)
        getcoordinates!(X, dh.grid, cellid)
        disassemble!(ae, a, udofs)

        reinit!(fv_u, X, fidx)

        # AD:
        #fill!(fμ, 0.0)
        #diffresult_μ = integrate_fμu!(diffresult_μ, fμ, fv_u, ip_μ, X, ae);
        #ke_μ2 = DiffResults.jacobian(diffresult_μ)

        # Analytical
        fill!(ke_μ, 0.0)
        fill!(fμ, 0.0)
        integrate_fμu_2!(ke_μ, fμ, fv_u, ip_μ, X, ae);

        assemble!(assembler_μ, μdofs, udofs, -ke_μ  * (1/A◫) ) #matrices.Kμu[μdofs, udofs] += -ke_μ  * (1/A◫) 

        if !rve.SOLVE_FOR_FLUCT
            fill!(fμ, 0.0)
            integrate_fμ_ext!(fμ, fv_u, ip_μ, X, macroscale.∇u, macroscale.∇w, macroscale.∇θ);
            matrices.fext_μ[μdofs] += fμ * (1/A◫)
        end
    end

    @info "Gamma -"
    for (cellid, fidx) in Γ⁻
        
        celldofs!(udofs, dh, cellid)
        getcoordinates!(X, dh.grid, cellid)        
        disassemble!(ae, a, udofs)
        
        reinit!(fv_u, X, fidx)
        
        # AD:
        #fill!(fμ, 0.0)
        #diffresult_μ = integrate_fμu!(diffresult_μ, fμ, fv_u, ip_μ, X, ae);
        #ke_μ2 = DiffResults.jacobian(diffresult_μ)
        
        # Analytical:
        fill!(ke_μ, 0.0)
        fill!(fμ, 0.0)
        integrate_fμu_2!(ke_μ, fμ, fv_u, ip_μ, X, ae);
        
        assemble!(assembler_μ, μdofs, udofs, ke_μ  * (1/A◫) ) #matrices.Kμu[μdofs, udofs] += ke_μ  * (1/A◫) 

        if !rve.SOLVE_FOR_FLUCT
            fill!(fμ, 0.0)
            integrate_fμ_ext!(fμ, fv_u, ip_μ, X, macroscale.∇u, macroscale.∇w, macroscale.∇θ);
            matrices.fext_μ[μdofs] += -fμ * (1/A◫)
        end
    end

    matrices.Kμu = sparse(assembler_μ.I, assembler_μ.J, assembler_μ.V, rve.nμdofs, rve.nudofs)#end_assemble(assembler_μ)
end

function solve_it!(rve::RVE, state::State)
    if rve.SOLVE_STYLE == SOLVE_FULL
        _solve_it_full!(rve::RVE, state::State)
    elseif rve.SOLVE_STYLE == SOLVE_SCHUR
        _solve_it_schur!(rve::RVE, state::State)
    end

    if rve.SOLVE_FOR_FLUCT
        state.a[1:rve.nudofs] .+= rve.matrices.uM
    end

end

function _solve_it_schur!(rve::RVE, state::State)

    (; dh, ch,  matrices)   = rve
    (; nudofs, nμdofs, nλdofs) = rve

    nλdofs = rve.nλdofs 
    nμdofs = rve.nμdofs 
    nλμdofs = nλdofs + nμdofs

    K = matrices.Kuu
    fext_u = matrices.fext_u
    RHS = zeros(Float64, nudofs, 1 + nλdofs + nμdofs)
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


    if rve.linearsolver === nothing
        LHS = K\RHS
    else
        LHS = zeros(size(RHS))

        @info "Solving Linear problem with schur compliment"
        @info "Preconditioner: $((rve.preconditioner))" 
        @info "Linear solver: $(typeof(rve.linearsolver))"

        print("Precon")
        @time precon = rve.preconditioner(K)
        prob   = LinearProblem(K, RHS[:,1])
        print("Linear cache")
        @time linsolve = LinearSolve.init(prob, rve.linearsolver, Pl = precon, verbose=true)

        for i in 1:size(RHS,2)
            @show i
            linsolve = LinearSolve.set_b(linsolve, RHS[:,i]) 
            print("Solving:::")
            @time sol = solve(linsolve)
            LHS[:,i] = sol.u
            #linsolve = sol.cache
        end
    end
    #@show Ferrite.LinearAlgebra.BLAS.set_num_threads(4)
    #@time LHS[:,1] .= K\RHS[:,1]
    #error("wait here")

    ub = LHS[:,1]               # ub =  K\fext_u
    Uλ = LHS[:,(1:(nλμdofs)) .+ 1] # Uλ = -K\Ct 

    ub[Ferrite.prescribed_dofs(ch)] .= 0.0 #apply_zero!(ub, ch)

    μλ = (Ct'*Uλ)\(-C*b - Ct'*ub + fμλ)

    state.a[(1:nudofs)           ] .= ub + Uλ*μλ
    state.a[(1:nλμdofs) .+ nudofs] .= μλ

    apply!(state.a, ch)
end

function _solve_it_full!(rve::RVE{dim}, state::State) where dim

    (; dh, ch, matrices)   = rve
    (; nμdofs, nλdofs) = rve

    @info "combining"

    KK = [matrices.Kuu matrices.Kμu' matrices.Kλu'; 
          matrices.Kμu spzeros(Float64, nμdofs, nμdofs) spzeros(Float64,nμdofs,nλdofs);
          matrices.Kλu spzeros(Float64, nλdofs, nμdofs) spzeros(Float64,nλdofs,nλdofs)]

    ff = vcat(matrices.fext_u, matrices.fext_μ, matrices.fext_λ)
    
    _update_this_to_ferrite_condense_sparsity_pattern!(KK, ch.acs)
    
    @info "applying"
    apply!(KK, ff, ch)

    if rve.linearsolver === nothing
        state.a .= KK\ff
    else

        @info "Solving Linear problem with full system"
        @info "Preconditioner: $((rve.preconditioner))" 
        @info "Linear solver: $(typeof(rve.linearsolver))"
    
        prob = LinearProblem(KK, ff)
        print("Precon:")
        @time precon = rve.preconditioner(KK)
        linsolve = LinearSolve.init(prob, rve.linearsolver, Pl = precon, verbose=true)

        print("Solve")
        @time sol = solve(linsolve)    
        state.a .= sol.u 
    end

    apply!(state.a, ch)
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

function evaluate_uᴹ!(uM::Vector{Float64}, rve::RVE{dim}, macroparameters::MacroParameters) where {dim}

    x̄ = zero(Vec{dim})

    chM = ConstraintHandler(rve.dh)
    add!(chM, Ferrite.Dirichlet(:u, Set(1:getnnodes(rve.grid)), (x, t) -> prolongation(x, x̄, macroparameters, rve.EXTRA_PROLONGATION), 1:dim))
    close!(chM)
    update!(chM, 0.0)
    apply!(uM, chM)

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


function _check_cellsets(parts, ncells::Int)

    all_cellsets = Int[]
    for part in parts
        for cellid in part.cellset
            if cellid in all_cellsets
                error("$cellid is in two sets")
            end
        end
        append!(all_cellsets, part.cellset)

        !issorted(part.cellset) && error("The cellset for the parts must be sorted.")
    end

    length(all_cellsets) < ncells && error("Not all cells are included in a part.")

end

function reset_constrainthandler!(ch::ConstraintHandler)
    empty!(ch.dbcs)
    empty!(ch.acs)
    empty!(ch.dbcs)
    empty!(ch.prescribed_dofs)
    empty!(ch.free_dofs)
    empty!(ch.inhomogeneities)
    empty!(ch.dofmapping)
    empty!(ch.bcvalues)
    ch.closed[] = false
end

include("response.jl")

export MacroParameters
export RVESubPartState
export State
#TODO: move to Ferrite?
function Ferrite.assemble!(a::Ferrite.Assembler{T}, rowdofs::AbstractVector{Int}, coldofs::AbstractVector{Int}, Ke::AbstractMatrix{T}) where {T}
    nrows = length(rowdofs)
    ncols = length(coldofs)

    @assert(size(Ke,1) == nrows)
    @assert(size(Ke,2) == ncols)

    append!(a.V, Ke)
    @inbounds for i in 1:ncols
        for j in 1:nrows
            push!(a.J, coldofs[i])
            push!(a.I, rowdofs[j])
        end
    end
end

end

