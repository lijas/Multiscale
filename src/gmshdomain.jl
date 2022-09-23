#http://jsdokken.com/converted_files/tutorial_gmsh.html

function addbox(a::AABB{2})
    mainbox = gmsh.model.occ.addRectangle(a.corner[1],a.corner[2], 0.0, a.lengths[1], a.lengths[2])
end

function addsphere(a::Inclusion{2})
    sphere = gmsh.model.occ.addDisk(a.pos[1],a.pos[2],0,a.radius,a.radius)
end

function addbox(a::AABB{3})
    mainbox = gmsh.model.occ.addBox(a.corner[1],a.corner[2], a.corner[3], a.lengths[1], a.lengths[2], a.lengths[3])
end

function addsphere(a::Inclusion{3})
    sphere = gmsh.model.occ.addSphere(a.pos[1],a.pos[2],a.pos[3], a.radius)
end

function trim_edges(all, domain::AABB{2})

    x = domain.corner[1]
    y = domain.corner[2]
    w = domain.lengths[1]
    h = domain.lengths[2]

    #Trim inclustions that are outside
    cw = max(w,h)*5
    cutleft  = gmsh.model.occ.addRectangle(x - cw,     y-cw/2, 0.0, cw, cw)
    cutright = gmsh.model.occ.addRectangle(x + w,      y-cw/2, 0.0, cw, cw)
    cuttop   = gmsh.model.occ.addRectangle(x,     y + h, 0.0, w, cw)
    cutbot   = gmsh.model.occ.addRectangle(x,     y - cw, 0.0, w, cw)
    for cut in [cutleft, cutright, cuttop, cutbot]
        all, _ = gmsh.model.occ.cut(all, [(2,cut)])
    end 

end

function trim_edges(all, domain::AABB{3})

    x = domain.corner[1]
    y = domain.corner[2]
    z = domain.corner[3]
    w = domain.lengths[1]
    d = domain.lengths[2]
    h = domain.lengths[3]

    #Trim inclustions that are outside
    cw = max(w, max(d, h))*5
    cutleft  = gmsh.model.occ.addBox(x - cw, y-cw/2, z - cw/2, cw, cw, cw)
    cutright = gmsh.model.occ.addBox(x + w , y-cw/2, z - cw/2, cw, cw, cw)

    cutfront   = gmsh.model.occ.addBox(x - cw/2, y - cw, z - cw/2, cw, cw, cw)
    cutback    = gmsh.model.occ.addBox(x - cw/2, y + d, z - cw/2, cw, cw, cw)

    cutbot   = gmsh.model.occ.addBox(x - cw/2, y - cw/2, z - cw, cw, cw, cw)
    cuttop   = gmsh.model.occ.addBox(x - cw/2, y - cw/2, z + h, cw, cw, cw)

    for cut in [cutleft, cutright, cuttop, cutbot, cutfront, cutback]
        all, _ = gmsh.model.occ.cut(all, [(3,cut)])
    end 

end

function get_entity_tag(a::Inclusion{2})
    eps = a.radius/100
    gmsh.model.getEntitiesInBoundingBox(a.pos[1] - a.radius - eps, a.pos[2] - a.radius - eps, -eps, 
                                        a.pos[1] + a.radius + eps, a.pos[2] + a.radius + eps, +eps, 2)
end

function get_entity_tag(a::Inclusion{3})
    eps = a.radius/100
    gmsh.model.getEntitiesInBoundingBox(a.pos[1] - a.radius - eps, a.pos[2] - a.radius - eps, a.pos[3] - a.radius - eps, 
                                        a.pos[1] + a.radius + eps, a.pos[2] + a.radius + eps, a.pos[3] + a.radius + eps, 3)
end

function get_domain_tag(a::AABB{2}, dim = 2)
    eps = a.lengths[1]/100
    gmsh.model.getEntitiesInBoundingBox(a.corner[1]-eps, a.corner[2]-eps, -eps, a.corner[1] + a.lengths[1] +eps, a.corner[2] + a.lengths[2]+eps, eps, dim)
end

function get_domain_tag(a::AABB{3})
    eps = a.lengths[1]/100
    gmsh.model.getEntitiesInBoundingBox(a.corner[1]-eps, a.corner[2]-eps, a.corner[3]-eps, a.corner[1] + a.lengths[1] + eps, a.corner[2] + a.lengths[2] + eps, a.corner[3] + a.lengths[3] + eps, 3)
end

function get_left_face(domain::AABB{2}, pad=domain.lengths[1]/100)
    x = domain.corner[1]-pad
    y = domain.corner[2]-pad
    z = -pad
    x2 = domain.corner[1]+pad
    y2 = domain.corner[2]+domain.lengths[2]+pad
    z2 = +pad

    gmsh.model.getEntitiesInBoundingBox(x, y, z, x2, y2, z2, 1)
end

function get_right_face(domain::AABB{2}, pad=domain.lengths[1]/100)
    x = domain.corner[1]+domain.lengths[1]-pad
    y = domain.corner[2]-pad
    z = -pad
    x2 = domain.corner[1]+domain.lengths[1]+pad
    y2 = domain.corner[2]+domain.lengths[2]+pad
    z2 = +pad

    gmsh.model.getEntitiesInBoundingBox(x, y, z, x2, y2, z2, 1)
end

function get_left_face(domain::AABB{3}, pad=domain.lengths[1]/100)
    x = domain.corner[1]-pad
    y = domain.corner[2]-pad
    z = domain.corner[3]-pad
    x2 = domain.corner[1]+pad
    y2 = domain.corner[2]+domain.lengths[2]+pad
    z2 = domain.corner[3]+domain.lengths[3]+pad

    gmsh.model.getEntitiesInBoundingBox(x, y, z, x2, y2, z2, 2)
end

function get_right_face(domain::AABB{3}, pad=domain.lengths[1]/100)
    x = domain.corner[1]+domain.lengths[1]-pad
    y = domain.corner[2]-pad
    z = domain.corner[3]-pad
    x2 = domain.corner[1]+domain.lengths[1]+pad
    y2 = domain.corner[2]+domain.lengths[2]+pad
    z2 = domain.corner[3]+domain.lengths[3]+pad

    gmsh.model.getEntitiesInBoundingBox(x, y, z, x2, y2, z2, 2)
end

function get_front_face(domain::AABB{3}, pad=domain.lengths[1]/100)
    x = domain.corner[1]-pad
    y = domain.corner[2]-pad
    z = domain.corner[3]-pad
    x2 = domain.corner[1]+domain.lengths[1]+pad
    y2 = domain.corner[2]+pad
    z2 = domain.corner[3]+domain.lengths[3]+pad

    gmsh.model.getEntitiesInBoundingBox(x, y, z, x2, y2, z2, 2)
end

function get_back_face(domain::AABB{3}, pad=domain.lengths[1]/100)
    x = domain.corner[1]-pad
    y = domain.corner[2]+domain.lengths[2]-pad
    z = domain.corner[3]-pad
    x2 = domain.corner[1]+domain.lengths[1]+pad
    y2 = domain.corner[2]+domain.lengths[2]+pad
    z2 = domain.corner[3]+domain.lengths[3]+pad
    gmsh.model.getEntitiesInBoundingBox(x, y, z, x2, y2, z2, 2)
end

function match_faces!(domain::AABB) 
    eps = domain.lengths[1]/100

    lx = domain.lengths[1]
    mainboxdimtags = get_domain_tag(domain)

    leftfacetags = get_left_face(domain, eps)
    rightfacetags = get_right_face(domain, eps)

    copyleftface = gmsh.model.occ.copy(leftfacetags)
    copyrightface = gmsh.model.occ.copy(rightfacetags)

    
    gmsh.model.occ.translate(copyleftface, lx, 0.0, 0.0)
    gmsh.model.occ.translate(copyrightface, -lx, 0.0, 0.0)

    gmsh.model.occ.synchronize()
    
    objecttags = union(mainboxdimtags, leftfacetags, rightfacetags)
    tooltags = union(copyleftface, copyrightface)

    gmsh.model.occ.fragment(objecttags, tooltags, -1, true,true)
    #append!(leftfacetags, mainboxdimtags)
    #gmsh.model.occ.fragment(leftfacetags, copyrightface, -1, true,true)
end

function get_new_tag(oldtag, alltags, map)

end

function generate_gmsh(sd::SampleDomain{dim}, meshsize::Float64) where dim

    gmsh.initialize()
    gmsh.clear()
    gmsh.initialize()

    gmsh.model.add("t16_2d")

    mainbox = addbox(sd.domain)
    lx = sd.domain.lengths[1]
    ly = sd.domain.lengths[2]

    #@show sd.domain
    #@show sd.inclusions

    @show mainbox  
    holes = []
    holetags = []
    for sphere in sd.inclusions
        sphere = addsphere(sphere)
        push!(holes, (dim, sphere))
        push!(holetags, sphere)
    end

    all, map = gmsh.model.occ.fragment([(dim,mainbox)], holes, -1, true,true)

    #mainboxtags = map[mainbox] #newtags for box

    #=
    println("fragment produced volumes:")
    for e in all
        println(e)
    end
    println("before/after fragment relations:")
    for e in zip(append!([(2, 1)],holes), map)
        println("parent " * string(e[1]) * " -> child " * string(e[2]))
    end=#
    
    #Trim inclustions that are outside
    trim_edges(all, sd.domain)

    gmsh.model.occ.synchronize()

    holetags = []
    for a in sd.inclusions
        newtag = get_entity_tag(a)
        append!(holetags, newtag)
    end

    #mainboxtags = get_domain_tag(sd.domain)
    eps = sd.domain.lengths[1]/100

    #leftfacetag = get_left_face(sd.domain, eps)
    #rightfacetag = get_right_face(sd.domain, eps)

    #push!(rightfacetag, (2,4))
    #copyleftface = gmsh.model.occ.copy(leftfacetag)
    #gmsh.model.occ.translate(copyleftface, lx, 0.0, 0.0)
    #gmsh.model.occ.fragment(rightfacetag, copyleftface, -1, true,true)

    match_faces!(sd.domain)
    gmsh.model.occ.synchronize()
    @show leftfacetags = get_left_face(sd.domain, eps)
    @show rightfacetags = get_right_face(sd.domain, eps)
    
    #gmsh.fltk.run()

    #getindex.(leftfacetag,2)
    translation = [1, 0, 0, -lx, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    for ltag in leftfacetags
        for rtag in rightfacetags
            try
                gmsh.model.mesh.setPeriodic(dim-1, [ltag[1]], [rtag[1]], translation)
            catch e
                println("ltag $(ltag) not working for rtag $(rtag)")
                continue
            end
        end
    end

    if dim == 4
        frontfacetag = get_front_face(sd.domain, eps)
        backfacetag = get_back_face(sd.domain, eps)
        translation = [1, 0, 0, 0, 0, 1, 0, -ly, 0, 0, 1, 0, 0, 0, 0, 1]
        try 
            gmsh.model.mesh.setPeriodic(dim-1, [frontfacetag[1][2]], [backfacetag[1][2]], translation)
        catch e
            throw(e)
        end
    end

    #@show newtag
    #@show holetags
    holetags = []
    for a in sd.inclusions
        newtag = get_entity_tag(a)
        append!(holetags, newtag)
    end

    grouptage1 = gmsh.model.addPhysicalGroup(dim, getindex.(holetags, 2))
    gmsh.model.setPhysicalName(dim, grouptage1, "inclusions")
    
    newtag = get_domain_tag(sd.domain)
    grouptage2 = gmsh.model.addPhysicalGroup(dim, getindex.(newtag, 2))
    gmsh.model.setPhysicalName(dim, grouptage2, "matrix")

    lcar1 = meshsize#sd.domain.lengths[end]/30

    ov = gmsh.model.getEntities(0);
    gmsh.model.mesh.setSize(ov, lcar1);

    #eps = 1e-3
    #ov = gmsh.model.getEntitiesInBoundingBox(x-eps, y-eps, -eps, x+eps, y+eps, +eps, 0)
    #gmsh.model.mesh.setSize(ov, lcar2)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim)

    gmsh.write("domainmesh.msh")
    #grid = saved_file_to_grid("domainmesh.msh")

    grid = getgrid()

    gmsh.finalize()

    return grid

end

function getgrid()

    dim = Int64(gmsh.model.getDimension())
    facedim = dim - 1

    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    nodes = FerriteGmsh.tonodes()
    elements, gmsh_elementidx = FerriteGmsh.toelements(dim)
    cellsets = FerriteGmsh.tocellsets(dim, gmsh_elementidx)

    boundarydict = FerriteGmsh.toboundary(2)
    facesets = FerriteGmsh.tofacesets(boundarydict, elements)

    return Grid(elements, nodes, facesets=facesets, cellsets=cellsets)
end