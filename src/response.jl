
function calculate_response(rve::RVE{dim}, state::State, AD = false) where dim

    N = Ref( zero(Tensor{2,dim-1,Float64}) )
    M = Ref( zero(Tensor{2,dim-1,Float64}) )
    V = Ref( zero(Vec{dim-1,Float64}) )

    _M = (1,3,6)[dim] #Yuck!
    cellstresses = Vector{Vector{SymmetricTensor{2,dim,Float64,_M}}}(undef, getncells(rve.grid)) 
    for (partid, part) in enumerate(rve.parts)
        material = part.material
        cellset = part.cellset
        matstates = state.partstates[partid].materialstates

        _calculate_response(N, V, M, cellstresses, rve, material, matstates, cellset, state.a, AD)
    end

    return N[], V[], M[], cellstresses
end

function _calculate_response(N::Ref{<:Tensor}, V::Ref{<:Vec}, M::Ref{<:Tensor}, cellstresses::Vector{<:Vector{<:SymmetricTensor{2,dim}}}, rve::RVE{dim}, material::AbstractMaterial, materialstates::Vector{Vector{MS}}, cellset::Vector{Int}, a::Vector{Float64}, AD) where {dim,MS<:AbstractMaterialState}

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

        _N,_V,_M, σ_cell = calculate_NVM(cv_u, X, ae, material, states, rve.EXTRA_PROLONGATION, AD)

        cellstresses[cellid] = σ_cell

        N[] += _N * (1/A◫)
        V[] += _V * (1/A◫)
        M[] += _M * (1/A◫)
    end

end

function calculate_NVM(cv::CellVectorValues, X::Vector{Vec{dim,Float64}}, ae::Vector{Float64}, material::AbstractMaterial, states, EXTRA_PROLONGATION, AD) where dim

    if AD
        _N,_V,_M, σ_cell = _calculate_NVM_AD(cv, X, ae, material, states, EXTRA_PROLONGATION)
    else
        _N,_V,_M, σ_cell = _calculate_NVM(cv, X, ae, material, states, EXTRA_PROLONGATION)
    end

end

function _calculate_NVM(cv::CellVectorValues, X::Vector{Vec{dim,Float64}}, ae::Vector{Float64}, material::AbstractMaterial, states, EXTRA_PROLONGATION) where dim

    N = zero(Tensor{2,dim,Float64})
    M = zero(Tensor{2,dim,Float64})
    V = zero(Vec{dim,Float64})
    
    e = basevec(Vec{dim,Float64})
    Î = (e[1]⊗e[1]) + (e[2]⊗e[2])
    x̄ = zero(Vec{dim,Float64})
    cellstresses = zeros(SymmetricTensor{2,dim,Float64,}, getnquadpoints(cv))
    for qp in 1:getnquadpoints(cv)

        dV = getdetJdV(cv, qp)
        x = spatial_coordinate(cv, qp, X)
        x̂ = Î ⋅ x
        z = x[dim]

        ε = symmetric( function_gradient(cv, qp, ae) )
        if dim == 2
            σ, _, _ = material_response(PlaneStrain(), material, ε, states[qp])
        else
            σ, _, _  = material_response(material, ε, states[qp])
        end
        cellstresses[qp] = σ

        N += (Î ⋅ σ ⋅ Î) * dV
        V += (Î⋅σ ⋅ e[dim]) * dV
        M += (z*(Î ⋅ σ ⋅ Î)) * dV
        if !EXTRA_PROLONGATION
            M += ((Î⋅σ ⋅ e[dim]) ⊗ (x̂ - x̄)) * dV
        else#EXTRA_PROLONGATION
            _tmp = ((Î⋅σ ⋅ e[dim]) ⊗ (x̂ - x̄))
            M +=  0.5*(_tmp - _tmp') * dV
        end

    end    
    _N = Tensor{2,dim-1,Float64}((i,j) -> N[i,j])
    _V = Vec{dim-1,Float64}((i) -> V[i])
    _M = Tensor{2,dim-1,Float64}((i,j) -> M[i,j])

    return _N, _V, _M, cellstresses
end

function _calculate_NVM_AD(cv::CellVectorValues, X::Vector{Vec{dim,Float64}}, ae::Vector{Float64}, material::AbstractMaterial, states, EXTRA_PROLONGATION) where dim

    e = basevec(Vec{dim,Float64})
    Î = (e[1]⊗e[1]) + (e[2]⊗e[2])
    x̄ = zero(Vec{dim,Float64})


    _N = zeros(Float64, dim-1, dim-1)
    _M = zeros(Float64, dim-1, dim-1)
    _V = zeros(Float64, dim-1)
    cellstresses = zeros(SymmetricTensor{2,dim,Float64,}, getnquadpoints(cv))
    for qp in 1:getnquadpoints(cv)

        dV = getdetJdV(cv, qp)
        x = spatial_coordinate(cv, qp, X)

        ε = symmetric( function_gradient(cv, qp, ae) )
        if dim == 2
            σ, _, _ = material_response(PlaneStrain(), material, ε, states[qp])
        else
            σ, _, _  = material_response(material, ε, states[qp])
        end
        cellstresses[qp] = σ

        for α in 1:(dim-1), β in 1:(dim-1)
            pertubations = MultiScale.MacroParameters{2}(
                ∇u = Tensor{2,2}( (i,j) -> (i==α && j==β ) ? 1.0 : 0.0)
            )

            δε = symmetric(gradient( x -> prolongation(x, x̄, pertubations, EXTRA_PROLONGATION), x))
            _N[α,β] += (σ ⊡ δε) * dV
        end


        for α in 1:(dim-1), β in 1:(dim-1)
            pertubations = MultiScale.MacroParameters{2}(
                ∇θ = Tensor{2,2}( (i,j) -> (i==α && j==β ) ? 1.0 : 0.0)
            )
            δε = symmetric(gradient( x -> prolongation(x, x̄, pertubations, EXTRA_PROLONGATION), x))
            _M[α,β] += (σ ⊡ δε) * dV
        end

        for α in 1:(dim-1)
            pertubations = MultiScale.MacroParameters{2}(
                ∇w = Vec{2}( (i) -> (i==α) ? 0.5 : 0.0),
                θ = Vec{2}( (i) -> (i==α) ? -0.5 : 0.0)
            )

            δε = symmetric(gradient( x -> prolongation(x, x̄, pertubations, EXTRA_PROLONGATION), x))
            #@show (σ ⊡ δε)
            _V[α] += (σ ⊡ δε) * dV
        end

        #for α in 1:(dim-1)
        #    pertubations = MultiScale.MacroParameters{2}(
        #        θ = Vec{2}( (i) -> (i==α) ? 1.0 : 0.0)
        #    )

        #    δε = symmetric(gradient( x -> prolongation(x, x̄, pertubations, EXTRA_PROLONGATION), x))
            #@show x
        #    _V[α] -= (σ ⊡ δε) * dV
        #end

    end

    N = Tensor{2,dim-1,Float64}((i,j) -> _N[i,j])
    V = Vec{dim-1,Float64}((i) -> _V[i])
    M = -Tensor{2,dim-1,Float64}((i,j) -> _M[i,j])

    return N, V, M, cellstresses
end



function check_asdf(rve::RVE{dim}, a_fluct) where {dim}

    (; dh, grid, cv_u) = rve

    nudofs = getnbasefunctions(cv_u)
    ae = zeros(Float64, nudofs)
    udofs = zeros(Int, nudofs)
    coords = zeros(Vec{dim,Float64}, Ferrite.nnodes_per_cell(dh.grid))

    residual1 = zero(Tensor{2,3,Float64,9})
    residual2 = zero(Vec{3,Float64})

    u◫ = zero(Vec{3})
    w◫ = zero(Float64)
    θ◫ = zero(Vec{3})
    h◫ = zero(Tensor{2,3})
    g◫ = zero(Vec{3})
    κ◫ = zero(Tensor{2,3})

    for cellid in 1:getncells(grid)
        celldofs!(udofs, dh, cellid)
        getcoordinates!(coords, dh.grid, cellid)
        disassemble!(ae, a_fluct, udofs)
        reinit!(cv_u, coords)

        for iqp in 1:getnquadpoints(cv_u)
            dV = getdetJdV(cv_u, iqp)
            us = function_value(cv_u, iqp, ae)
            xyz = spatial_coordinate(cv_u, iqp, coords)
            residual1 += us ⊗ xyz *dV
            residual2 += us * dV
        end
    
        Ω = rve.Ω◫
        I = rve.I◫

        u◫ += 1/Ω * MultiScale.u◫_operator(cv_u, ae)
        w◫ += 1/Ω * MultiScale.w◫_operator(cv_u, ae)
        θ◫ -= 1/I * MultiScale.θ◫_operator(cv_u, ae, coords)
        h◫ += 1/Ω * MultiScale.h◫_operator(cv_u, ae)
        g◫ += 1/Ω * MultiScale.g◫_operator(cv_u, ae)
        κ◫ -= 1/I * MultiScale.κ◫_operator(cv_u, ae, coords)

    end

    return u◫, w◫, θ◫, h◫, g◫, κ◫
end