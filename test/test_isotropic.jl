

function build_and_run(; dim::Int, L◫::Float64, h::Float64, macroscale::MultiScale.MacroParameters, material::AbstractMaterial, bctype::MultiScale.BCType, solvestyle::MultiScale.SolveStyle)
    
    elsize = 0.5

    nelx = round(Int, L◫/elsize)
    nelz = round(Int, h/elsize)
    nels =  (nelx, nelx, nelz) 
    corner = Vec{dim,Float64}( ( L◫/2, L◫/2, h/2 ) )
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

    rve = MultiScale.RVE(;
        grid, 
        parts = MultiScale.RVESubPart[
            MultiScale.RVESubPart(
                material = material,
                cellset = 1:getncells(grid) |> collect
            )],
        BC_TYPE = bctype,
        SOLVE_STYLE = solvestyle,
        PERFORM_CHECKS = true
    )

    state = State(rve)
    MultiScale.solve_rve(rve, macroscale, state)

    N,V,M = MultiScale.calculate_response(rve, state)


    return N,V,M 
end

@testset "Test Isotripic" begin

    dim = 3
    h = 5.0

    material = LinearElastic(E = 210.0, ν = 0.0 )

    macroscale = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((0.00, 0.00, 0.0, 0.00)), 
        ∇w = Vec{2}((0.0, 0.00)), 
        ∇θ = Tensor{2,2}((0.01, 0.00, 0.00, 0.00)), 
        w = 0.0, 
        θ = Vec{2}((0.0,0.0))
    )

    Ns = []
    Vs = []
    Ms = []
    
    N_plate, M_plate, V_plate = calculate_anlytical(material, macroscale, [0.0], [-h/2, h/2])

    Ls = [5.0, 10.0]#, 30.0]#, 4.0, 7.0]
    solvestyles = [MultiScale.SOLVE_FULL, MultiScale.SOLVE_FULL, MultiScale.SOLVE_SCHUR]
    bctypes = [RELAXED_DIRICHLET(), WEAK_PERIODIC(), STRONG_PERIODIC()]#, DIRICHLET(),]
    for (solvestyle, bctype) in zip(solvestyles, bctypes)
        for L in Ls
            println("Length: $L")
            N, V, M = build_and_run(dim=dim, L◫=L, h=h, macroscale=macroscale, material=material, bctype=bctype, solvestyle=solvestyle)
            push!(Ns, N)
            push!(Ms, M)
            push!(Vs, V)
            @test isapprox(M[1,1], M_plate[1,1], atol = 0.5)
        end
    end


    @show Ms

end
