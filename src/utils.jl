
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