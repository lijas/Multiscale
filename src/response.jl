
function calculate_response(rve::RVE{dim}, state::State) where dim

    N = Ref( zero(SymmetricTensor{2,dim-1,Float64}) )
    M = Ref( zero(Tensor{2,dim-1,Float64}) )
    V = Ref( zero(Vec{dim-1,Float64}) )
    

    for (partid, part) in enumerate(rve.parts)
        material = part.material
        cellset = part.cellset
        matstates = state.partstates[partid].materialstates

        _calculate_response(N, V, M, rve, material, matstates, cellset, state.a)
    end

    return N[], V[], M[]
end

function _calculate_response(N::Ref{<:SymmetricTensor}, V::Ref{<:Vec}, M::Ref{<:Tensor}, rve::RVE{dim}, material::AbstractMaterial, materialstates::Vector{Vector{MS}}, cellset::Vector{Int}, a::Vector{Float64}) where {dim,MS<:AbstractMaterialState}

    (; udofs, X, ae, diffresult_μ) = rve.cache
    (; dh, cv_u)    = rve
    (; diffresult_μ) = rve.cache
    (; A◫) = rve


    nudofs = getnbasefunctions(cv_u)
    ae = zeros(Float64, nudofs)
    udofs = zeros(Int, nudofs)
    X = zeros(Vec{dim,Float64}, Ferrite.nnodes_per_cell(dh.grid))

    for (localid, cellid) in enumerate(cellset)

        celldofs!(udofs, dh, cellid)
        getcoordinates!(X, dh.grid, cellid)
        disassemble!(ae, a, udofs)

        reinit!(cv_u, X)

        states = materialstates[localid]

        _N,_V,_M = calculate_NVM(cv_u, X, ae, material, states)

        N[] += _N * (1/A◫)
        V[] += _V * (1/A◫)
        M[] += _M * (1/A◫)
    end

end

function calculate_NVM(cv::CellVectorValues, X::Vector{Vec{dim,Float64}}, ae::Vector{Float64}, material::AbstractMaterial, states) where dim

    N = zero(SymmetricTensor{2,dim-1,Float64})
    M = zero(Tensor{2,dim-1,Float64})
    V = zero(Vec{dim-1,Float64})
    
    for qp in 1:getnquadpoints(cv)

        dV = getdetJdV(cv, qp)
        xyz = spatial_coordinate(cv, qp, X)
        xₚ  = Vec{dim-1}(i -> xyz[i])
        z = xyz[dim]
        ε   = symmetric( function_gradient(cv, qp, ae) )

        σ, _, _ = material_response(material, ε, states[qp])

        σₚ = SymmetricTensor{2,dim-1,Float64}((i,j) -> σ[i,j])
        σz = Vec{dim-1,Float64}((i) -> σ[i,dim])

        N += (σₚ) * dV
        V += (σz) * dV
        M += (z*σₚ + σz ⊗ xₚ) * dV

    end    

    return N, V, M
end