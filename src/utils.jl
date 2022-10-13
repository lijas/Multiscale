
# Similar to Ferrite._condense!(K, ch), but only add the non-zero entries to K (that arises from the condensation process)
function _update_this_to_ferrite_condense_sparsity_pattern!(K::SparseMatrixCSC{T}, acs::Vector{Ferrite.AffineConstraint{T}}) where T
    ndofs = size(K, 1)
    (length(acs) == 0) && return 
    # Store linear constraint index for each constrained dof
    distribute = Dict{Int,Int}(acs[c].constrained_dof => c for c in 1:length(acs))

    #Adding new entries to K is extremely slow, so create a new sparsity triplet for the condensed sparsity pattern
    N = length(acs)*2 # TODO: Better size estimate for additional condensed sparsity pattern.
    I = Int[]; resize!(I, N)
    J = Int[]; resize!(J, N)

    cnt = 0
    for col in 1:ndofs
        # Since we will possibly be pushing new entries to K, the field K.rowval will grow.
        # Therefor we must extract this before iterating over K
        range = nzrange(K, col)
        _rows = K.rowval[range]
        dcol = get(distribute, col, 0)
        if dcol == 0
            for row in _rows
                drow = get(distribute, row, 0)
                if drow != 0
                    ac = acs[drow]
                    for (d, _) in ac.entries
                        if !Ferrite._addindex_sparsematrix!(K, 0.0, d, col)
                            cnt += 1
                            Ferrite._add_or_grow(cnt, I, J, d, col)
                        end
                    end
                end
            end
        else
            for row in _rows
                drow = get(distribute, row, 0)
                if drow == 0
                    ac = acs[dcol]
                    for (d, _) in ac.entries
                        if !Ferrite._addindex_sparsematrix!(K, 0.0, row, d)
                            cnt += 1
                            Ferrite._add_or_grow(cnt, I, J, row, d)
                        end
                    end
                else
                    ac1 = acs[dcol]
                    for (d1, _) in ac1.entries
                        ac2 = acs[distribute[row]]
                        for (d2, _) in ac2.entries
                            if !Ferrite._addindex_sparsematrix!(K, 0.0, d1, d2)
                                cnt += 1
                                Ferrite._add_or_grow(cnt, I, J, d1, d2)
                            end
                        end
                    end
                end
            end
        end
    end

    resize!(I, cnt)
    resize!(J, cnt)

    # Use eps(T) so that the it does not affect the current values of the sparse matrix (I need to call _condense_sparsity_pattern() in other places of my code /Elias)
    V = fill(eps(T), length(I))
    K2 = sparse(I, J, V, ndofs, ndofs)

    K .+= K2

    return nothing
end



function search_nodepairs(grid::Grid{dim,C,T}, facepairs::Vector{<:Pair}, side_lengths::NTuple{dim,T}) where {dim,C,T}

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

function add_linear_constraints!(grid::Grid{dim}, ch::ConstraintHandler, nodedofs::Matrix{Int}, macroscale, nodepairs#=::Dict{Int,Int}=#, masternode::Int, EXTRA_PROLONGATION, SOLVE_FOR_FLUCT) where dim

    x̄ = zero(Vec{dim})

    for (nodeid_s, nodeid_m) in nodepairs
        
        xm = grid.nodes[nodeid_m].x
        xs = grid.nodes[nodeid_s].x
        
        x_jump = xs-xm
        
        ∇u = increase_dim(macroscale.∇u)
        ∇θ = increase_dim(macroscale.∇θ)
        ∇w = increase_dim(macroscale.∇w)

        e₃ = basevec(Vec{dim,Float64})[dim]

        z = xm[dim]

        b = zero(Vec{dim})
        if !SOLVE_FOR_FLUCT
            #b = ∇u⋅x_jump - z*∇θ⋅x_jump + ∇w⋅x_jump * e₃
            b = prolongation(xs, x̄, macroscale, EXTRA_PROLONGATION) - prolongation(xm, x̄, macroscale, EXTRA_PROLONGATION)
        end

        dof_s = nodedofs[:,nodeid_s]
        dof_l = nodedofs[:,nodeid_m]
        
        for d in 1:dim #ncomponents?
            lc = AffineConstraint(dof_s[d], [ (dof_l[d] => 1.0) ], b[d])
            add!(ch, lc)
        end
        #add!(ch, Dirichlet(:u, Set([nodeid_r]), (x,t)->b[dim], [dim]))

    end

    #Lock a master node
    #lock_masternode = -1
    #if length(masternodes) == 0
    #    @assert dim == 2
    #    lock_masternode = first(nodepairs)[2] #Pick the first dof on Γ+
    #else
    #    lock_masternode = first(masternodes)
    #end
    
    add!(ch, Ferrite.Dirichlet(:u, Set([masternode]), (x,t) -> zero(Vec{dim}), 1:dim))

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