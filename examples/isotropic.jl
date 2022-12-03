
using MultiScale
using Ferrite
using Tensors
using MaterialModels
#using TimerOutputs
#using Plots; plotly()

include("example_utils.jl")

function run_isotropic()

    dim = 3
    h = 1.0

    material = LinearElastic(E = 210.0, ν = 0.3 )

    macroscale = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((0.10, 0.30, 0.4, 0.06)), 
        ∇w = Vec{2}((0.1, 0.10)), 
        ∇θ = Tensor{2,2}((1.0, 0.1, 0.10, 0.2)), 
        w = 0.0, 
        θ = Vec{2}((0.0,0.0))
    )

    Ns = []
    Vs = []
    Ms = []
    
    Ls = [2.0]#, 10., 20.0]#, 30.0]#, 4.0, 7.0]
    for L in Ls
        println("Length: $L")
        N, V, M = build_and_run(dim=dim, L◫=L, h=h, macroscale=macroscale, material=material)
        push!(Ns, N)
        push!(Ms, M)
        push!(Vs, V)
    end

    N_plate, M_plate, V_plate = calculate_anlytical(material, macroscale, [0.0], [-h/2, h/2])
    
    nothing
    #=
    #Plot results 
    fig = [plot(reuse=false) for _ in 1:3] 
    for d1 in 1:dim-1
        _Vs = getindex.(Vs, d1)
        plot!(fig[3], Ls, _Vs, xlabel = "Length Lbox", ylabel = "Shear force", label = "V$d1", mark = :o)
        for d2 in 1:dim-1
            _Ns = getindex.(Ns, d1, d2)
            _Ms = getindex.(Ms, d1, d2)
            plot!(fig[1], Ls, _Ns, xlabel = "Length Lbox", ylabel = "Normal force", label = "N$d1$d2", mark = :o)
            plot!(fig[2], Ls, _Ms, xlabel = "Length Lbox", ylabel = "Moment", label = "M$d1$d2", mark = :o)
        end
    end

    if dim == 2
        plot!(fig[1], Ls, fill(EA_beam, length(Ls)), label = "Beam theory")
        plot!(fig[2], Ls, fill(EI_beam, length(Ls)), label = "Beam theory")
        plot!(fig[3], Ls, fill(KGA_beam, length(Ls)), label = "Beam theory")
    elseif dim == 3
        for d1 in 1:2, d2 in 1:2
            plot!(fig[1], Ls, fill(N_plate[d1,d2], length(Ls)), label = "Plate theory N$d1$d2")
            plot!(fig[2], Ls, fill(M_plate[d1,d2], length(Ls)), label = "Plate theory M$d1$d2")
        end
        for d1 in 1:2
            plot!(fig[3], Ls, fill(V_plate[d1], length(Ls)), label = "Plate theory V$d1")
        end
    end
    display(fig[1])
    display(fig[2])
    display(fig[3])=#


end

function build_and_run(; dim::Int, L◫::Float64, h::Float64, macroscale::MultiScale.MacroParameters, material::AbstractMaterial)
    #dim = 3
    #L◫ = 10.0
    #h = 2.0
    
    elsize = 0.5
    nelx = round(Int, L◫/elsize)
    nelz = round(Int, h/elsize)
    nels =  (nelx, nelx, nelz) 
    corner = Vec{dim,Float64}( ( L◫/2, L◫/2, h/2 ) )
    grid = generate_grid(dim == 2 ? Quadrilateral : Hexahedron, nels, -corner, corner)

    addnodeset!(grid, "cornerset", (x) -> x ≈ corner)
    if dim == 3
        addfaceset!(grid, "Γ⁺", union(getfaceset(grid, "right"), getfaceset(grid, "back"))) 
        addfaceset!(grid, "Γ⁻", union(getfaceset(grid, "left"), getfaceset(grid, "front")))
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
        BC_TYPE = MultiScale.STRONG_PERIODIC(),
        SOLVE_STYLE = MultiScale.SOLVE_SCHUR,
        SOLVE_FOR_FLUCT = false,
        EXTRA_PROLONGATION = false,

    )

    state = State(rve)
    MultiScale.solve_rve(rve, macroscale, state)

    N,V,M = MultiScale.calculate_response(rve, state, false)
    N_AD, V_AD, M_AD  = MultiScale.calculate_response(rve, state, true)

    
    @show N N_AD N ≈ N_AD
    @show V V_AD V ≈ V_AD
    @show M M_AD M ≈ M_AD


    vtk_grid("rve_isotropic_$(round(Int,L◫))", rve.dh) do vtk
        vtk_point_data(vtk, rve.dh, state.a[1:ndofs(rve.dh)])
        vtk_cellset(vtk, grid)
    end

    return N,V,M 
end

