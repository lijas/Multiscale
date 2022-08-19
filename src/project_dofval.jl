

function project_to_new_grid!(old_dh::DofHandler{dim}, a, new_dh::DofHandler{dim}, a_new) where dim
	@assert length(old_dh.field_names) == 1
    @assert length(new_dh.field_names) == 1
    
    #Get info about the field
    _cell = first(new_dh.grid.cells) #Random cell to gett cell type
    geom_ip = Ferrite.default_interpolation(typeof(_cell))
    field_ip = new_dh.field_interpolations[1]
    field_dim = new_dh.field_dims[1]
    
    #Create cellvalues contining the positions of the dofs
    coords = Ferrite.reference_coordinates(field_ip)
    weights = zeros(Float64, length(coords))
    qr = QuadratureRule{dim,RefCube,Float64}(weights, coords)
	cv = CellScalarValues(qr, field_ip, geom_ip)
    
    #Allocate some arrays, standard stuff
    coords = zeros(Vec{dim}, getnbasefunctions(geom_ip))
    celldofs = zeros(Int, ndofs_per_cell(new_dh))

    new_dh_dofs = Int[]             
    new_dh_dof_positions = Vec{3,Float64}[] #Coordinates of each dof
    dofvisited_hash = Dict{Int,Nothing}()

    #Eval each dof position in each cell
	for cellid in 1:getncells(new_dh.grid)
        celldofs!(celldofs, new_dh, cellid)
        getcoordinates!(coords, new_dh.grid, cellid)
		
        #Loop through each dof position in the cell
		for idofloc in 1:getnquadpoints(cv)
		    dof = celldofs[(idofloc-1)*field_dim + 1] #only use the first dof id if the field has multiple dims

            #Check if we have already visited this dof from another cell
		    if !haskey(dofvisited_hash, dof)
		       x = spatial_coordinate(cv, idofloc, coords)
               push!(new_dh_dofs, dof)
               push!(new_dh_dof_positions, x)
               dofvisited_hash[dof] = nothing
		    end
		end
	end

	peh = PointEvalHandler(old_dh.grid, new_dh_dof_positions)
    data = get_point_values(peh, old_dh, a)

    for i in 1:length(data)
        dof_value = data[i]
        dof = new_dh_dofs[i]
        for d in 1:field_dim
            a_new[dof + d - 1] = dof_value[d]
        end
    end

end


function test_grid_proj()

    old_grid = generate_grid(Hexahedron, (30,30,20))
    new_grid = generate_grid(Hexahedron, (80,80,20))

    old_dh = DofHandler(old_grid)
    push!(old_dh, :u, 3)
    close!(old_dh)

    new_dh = DofHandler(new_grid)
    push!(new_dh, :u, 3)
    close!(new_dh)

    old_a = rand(Float64, ndofs(old_dh))
    new_a = zeros(Float64, ndofs(new_dh))
     
    @time project_to_new_grid!(old_dh, old_a, new_dh, new_a)
    
    vtk_grid("old_grid", old_dh) do vtk
        vtk_point_data(vtk, old_dh, old_a)
    end

    vtk_grid("new_grid", new_dh) do vtk
        vtk_point_data(vtk, new_dh, new_a)
    end

end