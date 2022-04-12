
using MultiScale
using Ferrite
using Tensors
using MaterialModels
#using TimerOutputs
using Plots; plotly()

include("example_utils.jl")

function generate_problemdata(; L◫::Float64, dim::Int, material::MultiScale.TransverseIsotropy, angles, coords)

    #L◫ = 1.0
    nels_per_layer = 3
    nlayers = length(angles) 
    
    h = maximum(coords) - minimum(coords)
    elsize = 0.5
    
    nels = ntuple(i -> i==dim ? (nlayers*nels_per_layer) : round(Int, L◫/elsize), dim)
    corner = Vec{dim,Float64}( (i) -> i==dim ? h/2 : L◫/2 )
    grid = generate_grid(dim == 2 ? Quadrilateral : Hexahedron, nels, -corner, corner)

    cellmaterials = zeros(Int, getncells(grid))
    materialstates = Vector{Vector{MultiScale.TransverseIsotropyState}}()

    for cellid in 1:getncells(grid)

        push!(materialstates, AbstractMaterialState[])
        for ilay in 1:nlayers

            x = getcoordinates(grid, cellid)
            meanx = sum(x)/length(x)

            if  coords[ilay] < meanx[3] < coords[ilay+1]
                cellmaterials[cellid] = ilay

                a = Tensors.rotate(Vec((1.0,0.0,0.0)), Vec((0.0,0.0,1.0)), angles[ilay])
                append!(materialstates[cellid], [initial_material_state(material, a) for _ in 1:2^dim])
                break;
            end
        end

    end
    
    for ilay in 1:nlayers
        addcellset!(grid, "layer$ilay", findall(i-> i==ilay, cellmaterials))
    end

    return grid, materialstates
end

function run_transverslyisotropic()

    dim = 3
    h = 5.0

    macroscale = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((0.00, 0.00, 0.0, 0.00)), 
        ∇w = Vec{2}((0.0, 0.00)), 
        ∇θ = Tensor{2,2}((0.01, 0.00, 0.00, 0.00)), 
        w = 0.0, 
        θ = Vec{2}((0.0,0.0))
    )

    material = MultiScale.TransverseIsotropy(
        L₌ = 2*10e3,   #[GPA]
        L⊥ = 150e3,   
        M₌ = 700e3, 
        G⊥ = 100e3, 
        G₌ = 70e3,
    )

    coords = [-2.0, 0.0, 2.0]
    angles = deg2rad.([0, 90])


    Ns = []
    Vs = []
    Ms = []
    
    Ls = [1.0, 5.0, 10., 20.0]#, 30.0]#, 4.0, 7.0]
    for L in Ls
        println("Length: $L")
        N, V, M = build_and_run2(; dim=dim, L◫=L, macroscale=macroscale, material=material, angles, coords)
        push!(Ns, N)
        push!(Ms, M)
        push!(Vs, V)
    end

    N_plate, M_plate, V_plate = calculate_anlytical(material, macroscale, angles, coords)

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
    display(fig[3])


end

function build_and_run2(; dim::Int, L◫::Float64, macroscale::MultiScale.MacroParameters, material::AbstractMaterial, angles::Vector{Float64}, coords::Vector{Float64})
    #dim = 3
    #L◫ = 10.0
    #h = 2.0

    grid, materialstate = generate_problemdata(; L◫=L◫, dim=dim, material, angles, coords)


    rve = MultiScale.RVE(;
        grid, 
        parts = MultiScale.RVESubPart[
            MultiScale.RVESubPart(
                material = material,
                cellset = getcellset(grid, "layer1") |> collect
            ),
            MultiScale.RVESubPart(
                material = material,
                cellset = getcellset(grid, "layer2") |> collect
            )],
        BC_TYPE = MultiScale.WEAK_PERIODIC()
    )

    MS = eltype(eltype(materialstate))

    partstates = RVESubPartState[
        RVESubPartState{MS}( materialstate[collect(getcellset(grid, "layer1"))] ),
        RVESubPartState{MS}( materialstate[collect(getcellset(grid, "layer2"))] )
    ]

    state = State(rve, partstates)

    a = MultiScale.solve_rve(rve, macroscale, state)

    N,V,M = MultiScale.calculate_response(rve, state)

    addcellset!(grid, "Γ⁺", first.(getfaceset(grid, "Γ⁺")))
    addcellset!(grid, "Γ⁻", first.(getfaceset(grid, "Γ⁻")))
    vtk_grid("rve_newcode_$(round(Int,L◫))", rve.dh) do vtk
        vtk_point_data(vtk, rve.dh, a[1:ndofs(rve.dh)])
        vtk_cellset(vtk, grid)
    end

    return N,V,M 
end

