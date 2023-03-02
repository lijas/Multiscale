
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


    material = LinearElastic(E = 210.0, ν = 0.3 )

    rve = MultiScale.RVE(;
        grid, 
        parts = MultiScale.RVESubPart[
            MultiScale.RVESubPart(
                material = material,
                cellset = 1:getncells(grid) |> collect
            )],
        BC_TYPE = MultiScale.STRONG_PERIODIC(),
        LOCK_NODE = true,
    )

    state = State(rve)

    MultiScale._apply_macroscale!(rve, macroscale, state)
    MultiScale._assemble_volume!(rve, macroscale, state)
    MultiScale.evaluate_uᴹ!(rve.matrices.uM, rve, macroscale)

    #
    # FULL
    #
    state2 = State(rve)
    MultiScale._solve_it_full!(rve, state2)
    state2.a[1:rve.nudofs] .+= rve.matrices.uM

    _,_,Mfull, _ = MultiScale.calculate_response(rve, state2, false);

    #
    # SCHUR
    #
    state1 = State(rve)
    MultiScale._solve_it_schur!(rve, state1)
    state1.a[1:rve.nudofs] .+= rve.matrices.uM
    _,_,Mschur, _ = MultiScale.calculate_response(rve, state1, false);
    
    
    @test isapprox.(Mschur, Mfull, atol = 1e-4) |> all

end


@testset "Test schur" begin 
    macroscale = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((0.0, 0.0, 0.0, 0.0)), 
        ∇w = Vec{2}((0.0, 0.0)), 
        ∇θ = Tensor{2,2}((1.0, 0.0, 0.0, 0.0)), 
        w = 0.0, 
        θ = Vec{2}((0.0,0.0))
    )

    _build_rve(2.0, 2.0, macroscale)
end
