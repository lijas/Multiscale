
function _create_rve(; dim::Int, L◫::Float64, h::Float64)
    
    elsize = 0.5

    material = LinearElastic(E = 210.0, ν = 0.0 )

    macroscale = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((0.00, 0.00, 0.0, 0.00)), 
        ∇w = Vec{2}((0.0, 0.00)), 
        ∇θ = Tensor{2,2}((0.01, 0.00, 0.00, 0.00)), 
        w = 0.0, 
        θ = Vec{2}((0.0,0.0))
    )

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
        BC_TYPE = WEAK_PERIODIC(),
    )

    return rve
end

@testset "test integrals analytical vs AD" begin

    rve = _create_rve(; dim = 3, L◫ = 1.0, h = 1.0)
    cache = rve.cache
    material = rve.parts[1].material

    coords = Ferrite.reference_coordinates(Ferrite.default_interpolation(Hexahedron))
    
    n = getnbasefunctions(rve.cv_u)
    ae = rand(Float64, n)
    y  = zeros(Float64, n)
    mstates = [initial_material_state(material) for i in 1:getnquadpoints(rve.cv_u)]
    
    reinit!(rve.cv_u, coords)
    reinit!(rve.fv_u, coords, 1)

    #
    fill!(cache.fλ, 0.0)
    diffresult_λ = MultiScale.integrate_fλθ!(cache.diffresult_λ, cache.fλ, rve.cv_u, coords, ae, 1)
    ke = DiffResults.jacobian(diffresult_λ)

    fill!(cache.fλ, 0.0)
    fill!(cache.ke_λ, 0.0)
    MultiScale.integrate_fλθ_2!(cache.ke_λ, cache.fλ, rve.cv_u, coords, ae, 1); 

    @test (cache.ke_λ ≈ ke) |> all

    #
    fill!(cache.fu, 0.0)
    diffresult_u = MultiScale.integrate_fuu!(cache.diffresult_u, cache.fu, rve.cv_u, material, mstates, ae);
    ke = DiffResults.jacobian(cache.diffresult_u)
    
    fill!(cache.ke, 0.0)
    fill!(cache.fu, 0.0)
    MultiScale.integrate_fuu2!(cache.ke, cache.fu, rve.cv_u, material, mstates, ae);

    @test (cache.ke ≈ ke) |> all

    #
    diffresult_μ = MultiScale.integrate_fμu!(cache.diffresult_μ, cache.fμ, rve.fv_u, rve.ip_μ, coords, ae);
    ke_μ = DiffResults.jacobian(diffresult_μ)

    fill!(cache.ke_μ, 0.0)
    fill!(cache.fμ, 0.0)
    MultiScale.integrate_fμu_2!(cache.ke_μ, cache.fμ, rve.fv_u, rve.ip_μ, coords, ae);

    @test (cache.ke_μ ≈ ke_μ) |> all
end