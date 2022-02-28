#http://jsdokken.com/converted_files/tutorial_gmsh.html
using FerriteGmsh

gmsh.initialize()

function addbox(a::AABB{2})
    mainbox = gmsh.model.occ.addRectangle(a.corner[1],a.corner[2], 0.0, a.lengths[1], a.lengths[2])
end

function addsphere(a::Inclusion{2})
    sphere = gmsh.model.occ.addDisk(a.pos[1],a.pos[2],0,a.radius,a.radius)
end


function generate_gmsh(sd::SampleDomain{2})

    gmsh.model.add("t16_2d")

    x = sd.domain.corner[1]
    y = sd.domain.corner[2]
    w = sd.domain.lengths[1]
    h = sd.domain.lengths[2]
    mainbox = addbox(sd.domain)

    holes = []
    holetags = []
    for sphere in sd.inclusions
        sphere = addsphere(sphere)
        push!(holes, (2, sphere))
        push!(holetags, sphere)
    end


    all, map = gmsh.model.occ.fragment([(2,mainbox)], holes, -1, true,true)


    #Trim inclustions that are outside
    cw = w*5
    cutleft  = gmsh.model.occ.addRectangle(x - cw,     y-cw/2, 0.0, cw, cw)
    cutright = gmsh.model.occ.addRectangle(x + w,      y-cw/2, 0.0, cw, cw)
    cuttop   = gmsh.model.occ.addRectangle(x,     y + h, 0.0, w, cw)
    cutbot   = gmsh.model.occ.addRectangle(x,     y - cw, 0.0, w, cw)
    for cut in [cutleft, cutright, cuttop, cutbot]
        all, _ = gmsh.model.occ.cut(all, [(2,cut)])
    end 

    gmsh.model.occ.synchronize()

    holetags = []
    eps = w/1000
    for a in sd.inclusions
        newtag = gmsh.model.getEntitiesInBoundingBox(a.pos[1] - a.radius - eps, a.pos[2] - a.radius - eps, -eps, 
                                                     a.pos[1] + a.radius + eps, a.pos[2] + a.radius + eps, +eps, 2)
        append!(holetags, newtag)
    end

    a = sd.domain
    newtag = gmsh.model.getEntitiesInBoundingBox(a.corner[1]-eps, a.corner[2]-eps, -eps, a.corner[1] + a.lengths[1] +eps, a.corner[2] + a.lengths[2]+eps, eps, 2)

    @show newtag
    @show holetags
    grouptage1 = gmsh.model.addPhysicalGroup(2, getindex.(holetags, 2))
    gmsh.model.setPhysicalName(2, grouptage1, "inclusions")
    
    grouptage2 = gmsh.model.addPhysicalGroup(2, getindex.(newtag, 2))
    gmsh.model.setPhysicalName(2, grouptage2, "matrix")

    lcar1 = w/30

    ov = gmsh.model.getEntities(0);
    gmsh.model.mesh.setSize(ov, lcar1);

    #eps = 1e-3
    #ov = gmsh.model.getEntitiesInBoundingBox(x-eps, y-eps, -eps, x+eps, y+eps, +eps, 0)
    #gmsh.model.mesh.setSize(ov, lcar2)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    #gmsh.write("domainmesh.msh")
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

    nodes = tonodes()
    elements, gmsh_elementidx = toelements(dim)
    cellsets = tocellsets(dim, gmsh_elementidx)

    boundarydict = toboundary(2)
    facesets = tofacesets(boundarydict, elements)

    return Grid(elements, nodes, facesets=facesets, cellsets=cellsets)
end