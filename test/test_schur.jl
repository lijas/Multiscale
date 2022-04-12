
function _build_rve(L◫, h, macroscale::MultiScale.MacroParameters{dimm1}) where dimm1

    dim = dimm1+1

    elsize = 0.1

    nelx = round(Int, L◫/elsize)
    nelz = round(Int, h/elsize)
    nels =  ntuple(d-> d==dim ? nelz : nelx, dim) 
    corner = Vec{dim,Float64}( d-> d == dim ? h/2 : L◫/2 )
    grid = generate_grid(dim == 2 ? Quadrilateral : Hexahedron, nels, -corner, corner)

    #Rename facesets
    addnodeset!(grid, "cornerset", (x) -> x ≈ corner)
    addnodeset!(grid, "right", MultiScale.faceset_to_nodeset(grid, getfaceset(grid, "right")))
    addnodeset!(grid, "left",  MultiScale.faceset_to_nodeset(grid, getfaceset(grid, "left")))
    if dim == 3
        addnodeset!(grid, "front", MultiScale.faceset_to_nodeset(grid, getfaceset(grid, "front")))
        addnodeset!(grid, "back",  MultiScale.faceset_to_nodeset(grid, getfaceset(grid, "back")))
        addfaceset!(grid, "Γ⁺", union(getfaceset(grid, "right"), getfaceset(grid, "front"))) 
        addfaceset!(grid, "Γ⁻", union(getfaceset(grid, "left"), getfaceset(grid, "back")))
    elseif dim == 2
        addfaceset!(grid, "Γ⁺", getfaceset(grid, "right")) 
        addfaceset!(grid, "Γ⁻", getfaceset(grid, "left"))
    end


    material = LinearElastic(E = 210.0, ν = 0.0 )

    rve = MultiScale.RVE(;
        grid, 
        parts = MultiScale.RVESubPart[
            MultiScale.RVESubPart(
                material = material,
                cellset = 1:getncells(grid) |> collect
            )],
        BC_TYPE = MultiScale.STRONG_PERIODIC()
    )

    state = State(rve)

    MultiScale._apply_macroscale!(rve, macroscale, state)
    MultiScale._assemble_volume!(rve, macroscale, state)
    
    state2 = State(rve)
    MultiScale._solve_it_full!(rve, state2)
    
    state1 = State(rve)
    MultiScale._solve_it_schur!(rve, state1)
    
    
    @test isapprox(norm(state1.a), norm(state2.a), atol = 1e-4)

    #N,V,M = MultiScale.calculate_response(rve, state)

    #return N,V,M 

end


@testset "Test schur" begin 
    macroscale = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((0.0, 0.0, 0.0, 0.0)), 
        ∇w = Vec{2}((0.01, 0.0)), 
        ∇θ = Tensor{2,2}((0.0, 0.0, 0.0, 0.0)), 
        w = 0.0, 
        θ = Vec{2}((0.0,0.0))
    )

    _build_rve(2.0, 2.0, macroscale)
end
