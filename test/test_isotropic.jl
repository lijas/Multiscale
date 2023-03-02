

function build_and_run(;  elsize, dim::Int, L◫::Float64, h::Float64, macroscale::MultiScale.MacroParameters, material::AbstractMaterial, bctype::MultiScale.BCType, )
    
    nelx = round(Int, L◫/elsize)
    nelz = round(Int, h/elsize)
    nels =  (nelx, nelx, nelz) 
    corner = Vec{dim,Float64}( ( L◫/2, L◫/2, h/2 ) )
    grid = generate_grid(dim == 2 ? Quadrilateral : Hexahedron, nels, -corner, corner)

    #Rename facesets
    addnodeset!(grid, "cornerset", (x) -> x ≈ corner)

    rve = MultiScale.RVE(;
        grid, 
        parts = MultiScale.RVESubPart[
            MultiScale.RVESubPart(
                material = material,
                cellset = 1:getncells(grid) |> collect
            )],
        BC_TYPE = bctype,
        SOLVE_STYLE = MultiScale.SOLVE_FULL,
        EXTRA_PROLONGATION = false,
        LOCK_NODE = false
    )

    state = State(rve)
    MultiScale.solve_rve(rve, macroscale, state)

    N,V,M, cellstresses = MultiScale.calculate_response(rve, state, false)
    N_AD, V_AD, M_AD, cellstresses  = MultiScale.calculate_response(rve, state, true)

    a_fluct = state.a[1:rve.nudofs] - rve.matrices.uM
    
    u◫, w◫, θ◫, h◫, g◫, κ◫, xcenter, zcenter = MultiScale.run_checks(rve, a_fluct)

    @test isapprox.(u◫, 0.0, atol = 1e-10) |> all
    @test isapprox(w◫, 0.0, atol = 1e-10)
    @test isapprox.(θ◫, 0.0, atol = 1e-10) |> all
    @test isapprox.(h◫, 0.0, atol = 1e-10) |> all
    @test isapprox.(g◫, 0.0, atol = 1e-10) |> all
    @test isapprox.(κ◫, 0.0, atol = 1e-10) |> all
    @test isapprox(xcenter, 0.0, atol = 1e-10) 
    @test isapprox(zcenter, 0.0, atol = 1e-10) 

    @test isapprox.(N, N_AD, atol = 1e-10) |> all
    @test isapprox.(V, V_AD, atol = 1e-10) |> all
    @test isapprox.(M, M_AD, atol = 1e-10) |> all


    return N,V,M 
end

@testset "Test Isotripic" begin

    dim = 3
    h = 5.0

    material = LinearElastic(E = 210.0, ν = 0.3 )
    mbend = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((0.00, 0.00, 0.0, 0.00)), 
        ∇w = Vec{2}((0.0, 0.00)), 
        ∇θ = Tensor{2,2}((1.00, 0.00, 0.00, 0.00)), 
        w = 0.0, 
        θ = Vec{2}((0.0,0.0))
    )

    mtension = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((1.00, 0.00, 0.0, 0.00)), 
        ∇w = Vec{2}((0.0, 0.00)), 
        ∇θ = Tensor{2,2}((0.00, 0.00, 0.00, 0.00)), 
        w = 0.0, 
        θ = Vec{2}((0.0,0.0))
    )

    mtransshear = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((0.00, 0.00, 0.0, 0.00)), 
        ∇w = Vec{2}((1.0, 0.00)), 
        ∇θ = Tensor{2,2}((0.00, 0.00, 0.00, 0.00)), 
        w = 0.0, 
        θ = Vec{2}((0.0,0.0))
    )

    mtwist = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((0.00, 0.00, 0.0, 0.00)), 
        ∇w = Vec{2}((0.0, 0.00)), 
        ∇θ = Tensor{2,2}((0.00, 1.00, 1.00, 0.00)), 
        w = 0.0, 
        θ = Vec{2}((0.0,0.0))
    )


    #Just meassure change:
    N_plate, M_plate, V_plate = calculate_anlytical(material, mbend, [0.0], [-h/2, h/2])
    N, V, M = build_and_run(elsize = 0.05, dim=dim, L◫=0.5, h=h, macroscale=mbend, material=material, bctype=RELAXED_DIRICHLET(),)
    @test isapprox(M[1,1], (M_plate[1,1]), atol = 1e-0)

    N_plate, M_plate, V_plate = calculate_anlytical(material, mtension, [0.0], [-h/2, h/2])
    N, V, M = build_and_run(elsize = 0.5, dim=dim, L◫=5.0, h=h, macroscale=mtension, material=material, bctype=WEAK_PERIODIC(),)
    @test isapprox(N[1,1], (N_plate[1,1]), atol = 1e-5)

    N_plate, M_plate, V_plate = calculate_anlytical(material, mtransshear, [0.0], [-h/2, h/2])
    N, V, M = build_and_run(elsize = 0.05, dim=dim, L◫=0.5, h=h, macroscale=mtransshear, material=material, bctype=STRONG_PERIODIC(),)
    @test isapprox(V[1], (V_plate[1]*5/6), atol = 1e-1)

    N_plate, M_plate, V_plate = calculate_anlytical(material, mtwist, [0.0], [-h/2, h/2])
    N, V, M = build_and_run(elsize = 0.5, dim=dim, L◫=5.0, h=h, macroscale=mtwist, material=material, bctype=ENRICHED_PERIODIC(),)
    @test isapprox(M[1,2], M_plate[1,2], atol = 1e-8)

end