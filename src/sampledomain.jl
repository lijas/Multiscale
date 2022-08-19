export AABB, SampleDomain, SphericalInclusion
export generate_random_domain, cutout_inplane_subdomain, get_qp_domaintags
export plotdomain, plotdomain!, plotdomain_topview!, plotdomain_sideview!

#Used for contact detection of cylinders
using SolidModeling: SolidModeling, Solid, bintersect

abstract type AbstractInclusion{dim} end

const INCLUSION=1 
const MATRIX=2

struct AABB{dim}
    corner::Vec{dim, Float64}
    lengths::Vec{dim, Float64}
end

offset_aabb(a::AABB{dim}, offset::Vec{dim}) where dim = AABB(a.corner - offset, a.lengths)

minx(a::AABB) = a.corner[1]
miny(a::AABB) = a.corner[2]
minz(a::AABB) = a.corner[3]
maxx(a::AABB) = a.corner[1] + a.lengths[1]
maxy(a::AABB) = a.corner[2] + a.lengths[2]
maxz(a::AABB) = a.corner[3] + a.lengths[3]

height(a::AABB{dim}) where dim = a.lengths[dim]
side_length(a::AABB; dim::Int) = a.lengths[dim]

mincoord(a::AABB; dim::Int) = a.corner[dim]
maxcoord(a::AABB; dim::Int) = a.corner[dim] + a.lengths[dim]
volume(a::AABB) = prod(a.lengths)::Float64

#
# SPHERE
#

struct SphericalInclusion{dim} <: AbstractInclusion{dim}
    radius::Float64
    pos::Vec{dim,Float64}
    function SphericalInclusion(radius::Float64, pos::Vec{dim,Float64}) where dim
        @assert radius > 0.0
        return new{dim}(radius, pos)
    end
end

volume(i::SphericalInclusion{3}) = (4/3) * π * i.radius^3
volume(i::SphericalInclusion{2}) = π*i.radius^2

struct SampleDomain{dim}
    inclusions::Vector{<:AbstractInclusion{dim}}
    domain::AABB{dim}
end

function generate_random_inclusion(::Type{SphericalInclusion{dim}}, (radius_μ, radius_σ), pos::Vec{dim}) where dim
    data = Uniform(radius_μ - radius_σ, radius_μ + radius_σ)
    r = rand(data)
    return SphericalInclusion(r, pos)
end

function offset_inclusion(s::SphericalInclusion{dim}, offset::Vec{dim}) where {dim}
    return SphericalInclusion(s.radius, s.pos - offset)
end

point_inside_inclusion(x::Vec, s::SphericalInclusion) =  norm(s.pos - x) ≤ s.radius

#
# Cylinder
#

struct CylindricalInclusion{dim} <: AbstractInclusion{dim}
    radius::Float64
    length::Float64
    pos::Vec{dim,Float64} #Pos is the center of the fibre
    dir::Vec{dim,Float64}
    
    #Use a hexahedron from the Package SolidModeling for simplified contact search
    #Contact detection for cylinders was a bit tricky to implement...
    contact_hexahedron::SolidModeling.Solid

    function CylindricalInclusion(radius::Float64, length::Float64, pos::Vec{dim,Float64}, dir::Vec{dim,Float64}) where dim
        @assert radius > 0.0
        @assert length > 0.0
        @assert norm(dir) ≈ 1.0
        contact_hexahedron = create_hexahedron(radius,length,pos,dir)
        return new{dim}(radius, length, pos, dir, contact_hexahedron)
    end
end

volume(i::CylindricalInclusion{3}) = π * i.radius^2 * i.length
volume(i::CylindricalInclusion{2}) = i.length*i.radius*2

function generate_random_inclusion(::Type{CylindricalInclusion{dim}}, (radius_μ, radius_σ, length_μ, length_σ), pos::Vec{dim}) where {dim}
    data_r = Uniform(radius_μ - radius_σ, radius_μ + radius_σ)
    data_l = Uniform(length_μ - length_σ, length_μ + length_σ)
    r = rand(data_r)
    l = rand(data_l)

    #Create the fiber pointing in a random direction in-plane
    dir = Vec{3}((rand(), rand(), 0.0))
    dir /= norm(dir)

    if dim == 2
        dir = Vec((1.0,0.0))
    end

    return CylindricalInclusion(r, l, pos, dir)
end


function offset_inclusion(c::CylindricalInclusion{dim}, offset::Vec{dim}) where {dim}
    return CylindricalInclusion(c.radius, c.length, c.pos - offset, c.dir)
end

function point_inside_inclusion(pos::Vec, c::CylindricalInclusion{2}) 
    x = c.pos[1] - c.length/2
    y = c.pos[2] - c.radius
    w = c.length
    h = c.radius*2

    return (x < pos[1] < x + w) && (y < pos[2] < y+h)
end

function point_inside_inclusion(X::Vec, c::CylindricalInclusion{3}) 
    t1 = (X - c.pos)⋅c.dir

    if t1 > -c.length/2 && t1 < c.length/2     
        Q1 = c.pos + t1*c.dir 
        dir = X - Q1
        dist = norm(dir)

        if dist < c.radius
            return true
        end
    end

    return false
end


function offset_domain(sd::SampleDomain, offset)

    new_spheres = similar(sd.inclusions)

    for i in 1:length(sd.inclusions) 
        new_spheres[i] = offset_inclusion(sd.inclusions[i], offset)
    end

    return SampleDomain( new_spheres, offset_aabb(sd.domain, offset))
end

function volumefraction(sd::SampleDomain)
    volume_inclusions = 0.0
    for inclu in sd.inclusions
        volume_inclusions += volume(inclu)
    end
    return volume_inclusions / volume(sd.domain)
end

function generate_random_domain(
                    InclusionType::Type{T}, 
                    inclusion_para, 
                    aabb::AABB{dim}, 
                    ninclusions::Int; 
                    max_ntries::Int = ninclusions*10,
                    allow_outside_domain::Bool = false, 
                    allow_collisions::Bool = false
    ) where {dim, T<:AbstractInclusion{dim}}

    inclusions = InclusionType[]
    
    udpos = ntuple(d -> Uniform( mincoord(aabb, dim=d), maxcoord(aabb, dim=d)), dim)
    
    nadded = 0
    ntries = 0
    while nadded < ninclusions
        pos = generate_random_position(udpos)
        new_inclusion = generate_random_inclusion(InclusionType, inclusion_para, pos)

        should_not_add_inclusion =  (allow_collisions     ? false : collides_with_other_inclusion(inclusions, new_inclusion)) || 
                                    (allow_outside_domain ? false : inclusion_outside_domain(aabb, new_inclusion))

        if should_not_add_inclusion
            if ntries > max_ntries
                error("Could not distribute all voids: $nadded/$ninclusions")
            end
            ntries += 1
            continue
        else
            nadded += 1
            @info "Added inclusion: $(nadded)/$ninclusions"
            push!(inclusions, new_inclusion)
        end
        

    end

    return SampleDomain(inclusions, aabb)
end

generate_random_position(uniform::NTuple{dim, Uniform}) where dim = Vec{dim,Float64}( (d) -> rand(uniform[d]) )


function collides_with_other_inclusion(inclusions::Vector{SphericalInclusion{dim}}, new_inclusion::SphericalInclusion) where dim

    for inclusion_check in inclusions
        dist = norm(inclusion_check.pos - new_inclusion.pos)
        tmp = inclusion_check.radius + new_inclusion.radius
        check = dist <= tmp
        if check
            return true
        end
    end

    return false
end

function collides_with_other_inclusion(inclusions::Vector{CylindricalInclusion{2}}, ni::CylindricalInclusion)
    @assert abs(ni.dir ⋅ Vec((1.0,0.0))) ≈ 1.0

    i2x = ni.pos[1] - ni.length/2
    i2y = ni.pos[2] - ni.radius
    i2w = ni.length
    i2h = ni.radius*2

    for i1 in inclusions
        
        i1x = i1.pos[1] - i1.length/2
        i1y = i1.pos[2] - i1.radius
        i1w = i1.length
        i1h = i1.radius*2

        collision =    (i1x < i2x + i2w &&
                        i1x + i1w > i2x &&
                        i1y < i2y + i2h &&
                        i1y + i1h > i2y)

        if collision
            return true
        end
    end

    return false
end

function collides_with_other_inclusion(inclusions::Vector{CylindricalInclusion{3}}, ni::CylindricalInclusion{3})

    i2x = ni.pos[1] - ni.length/2
    i2y = ni.pos[2] - ni.radius
    i2w = ni.length
    i2h = ni.radius*2

    for i1 in inclusions
        if _collides2(i1, ni)
            return true
        end 
    end

    return false
end

#Contact algorithm for Hexahedrons, not cylinders
function _collides2(a::CylindricalInclusion{3}, b::CylindricalInclusion{3})
    output = bintersect(a.contact_hexahedron, b.contact_hexahedron)
    return length(output.vertices) > 0
end

#Note there are edge cases where contact is not detected.
function _collides(a::CylindricalInclusion{3}, b::CylindricalInclusion{3})
    #https://math.stackexchange.com/questions/1993953/closest-points-between-two-lines

    offset_radius = a.radius*0.2
    offset_length_a = a.length*0.1
    offset_length_b = b.length*0.1

    #Should work for any orientation with small modification, but check that fibres are only in-plane
    @assert Tensors.cross(b.dir, a.dir)[1] ≈ 0.0
    @assert Tensors.cross(b.dir, a.dir)[2] ≈ 0.0

    dir3 = Tensors.cross(b.dir, a.dir)

    #special cas: Check if parallel
    if norm(dir3) ≈ 0.0
        dir = a.dir

        aξ = a.pos ⋅ dir
        bξ = b.pos ⋅ dir
        a1 = aξ - a.length/2
        a2 = aξ + a.length/2
        b1 = bξ - b.length/2
        b2 = bξ + b.length/2

        #Check overlap in fiber directions
        if a1 <= b2 && b1 <= a2

            #Check overlap of radius
            #Get closest point between cylinders
            aξ = (b.pos - a.pos) ⋅ dir
            cp = a.pos + aξ*dir
            
            dist = norm(cp - b.pos)
            if dist < (a.radius+b.radius)
                return true
            end
        end
        return false
    end

    A = zeros(Float64, 3, 3)
    A[:,1] .= a.dir
    A[:,2] .= -b.dir
    A[:,3] .= dir3
    t1,t2,t3 = A\(b.pos - a.pos)

    Q1 = a.pos + t1*a.dir
    Q2 = b.pos + t2*b.dir
    dist = t3 * norm(dir3)

    if (t1 < -a.length/2 - offset_length_a || t1 > a.length/2 + offset_length_a)
        return false
    end

    if (t2 < -b.length/2 + offset_length_b || t2 > b.length/2 + offset_length_b)
        return false
    end

    #@show dist
    #@show dist = norm(Q1 - Q2)
    if dist < (a.radius + b.radius + offset_radius)
        return true
    end

    return false
end

function inclusion_outside_domain(aabb::AABB{dim}, a::SphericalInclusion{dim}) where dim

    for d in 1:dim
        if (a.pos[d] - a.radius) < mincoord(aabb, dim = d)
            return true
        elseif (a.pos[d] + a.radius) > maxcoord(aabb, dim = d)
            return true
        end
    end
    return false
end

function inclusion_outside_domain(aabb::AABB{dim}, a::CylindricalInclusion{2}) where dim

    (a.pos[2] - a.radius) < mincoord(aabb, dim = 2) && return true
    (a.pos[2] + a.radius) > maxcoord(aabb, dim = 2) && return true
    
    #Does not matter if x-direction is outside domain
    #(a.pos[1] - a.length/2) < mincoord(aabb, dim = 1) && return true
    #(a.pos[1] + a.length/2) > maxcoord(aabb, dim = 1) && return true
    
    return false
    
end

function cutout_subdomain(sd::SampleDomain{dim}, L◫::Float64) where dim
    udpos = ntuple(d -> Uniform( mincoord(sd.domain, dim=d), maxcoord(sd.domain, dim=d)), dim)

    h = L◫
    size = Vec{dim,Float64}( d -> d==dim ? h : L◫ )
    x = generate_random_position(udpos)
   
    sub_aabb = AABB(x - 0.5*size , size)
    subdomain = SampleDomain(typeof(sd.inclusions), sub_aabb)

    for sphere in sd.inclusions
        if inclusion_inside_aabb(sphere, sub_aabb)
            push!(subdomain.inclusions, sphere)
        end
    end
    @show length(subdomain.inclusions)
    return subdomain
end

function cutout_inplane_subdomain(sd::SampleDomain{dim}, L◫::Float64) where dim
    InclusionType = eltype(sd.inclusions)

    h = height(sd.domain)
    udpos = ntuple(d -> Uniform( mincoord(sd.domain, dim=d), maxcoord(sd.domain, dim=d) - L◫), dim-1)

    size  = Vec{dim}( d -> d==dim ? h : L◫ )
    coord = Vec{dim}( d -> d!=dim ? rand(udpos[d]) : mincoord(sd.domain; dim=d) )
   
    sub_aabb = AABB(coord , size)
    subdomain = SampleDomain(InclusionType[], sub_aabb)

    for incl in sd.inclusions
        if inclusion_inside_aabb(incl, sub_aabb)
            push!(subdomain.inclusions, incl)
        end
    end
    
    return subdomain
end

function inclusion_inside_aabb(sphere::SphericalInclusion{dim}, aabb::AABB{dim}) where dim

    # Get closest point
    closest_point = Vec{dim,Float64}( (d) -> begin
            mincoord(aabb, dim=d) ≥ sphere.pos[d] ? 
                mincoord(aabb, dim=d) : (maxcoord(aabb, dim=d) ≤ sphere.pos[d]) ? 
                maxcoord(aabb, dim=d) : sphere.pos[d]
        end
    )

    dist = (sphere.pos - closest_point)
    dist_squared = dist⋅dist;

    return dist_squared < sphere.radius^2;
end

function inclusion_inside_aabb(c::CylindricalInclusion{2}, aabb::AABB{2})

    i2x = c.pos[1] - c.length/2
    i2y = c.pos[2] - c.radius
    i2w = c.length
    i2h = c.radius*2
        
    i1x = aabb.corner[1]
    i1y = aabb.corner[2]
    i1w = aabb.lengths[1]
    i1h = aabb.lengths[2]

    collision =    (i1x < i2x + i2w &&
                    i1x + i1w > i2x &&
                    i1y < i2y + i2h &&
                    i1y + i1h > i2y)

    return collision
end


function get_qp_domaintags(grid::Grid{dim}, sd::SampleDomain{dim}, cv::Ferrite.Values) where dim

    #The sample domain and grid might not be locateted in same place in space
    #Offset the sample domain...
    corner = Vec{dim}( d -> minimum(n->n.x[d], grid.nodes) )

    offset = sd.domain.corner - corner

    sd = offset_domain(sd, offset)

    #id = default_cellvalues(grid)
    cellquaddomains = zeros(Int, getnquadpoints(cv), getncells(grid))
    for cellid in 1:getncells(grid)
        coords = getcoordinates(grid, cellid)
        for iqp in 1:getnquadpoints(cv)
            x = spatial_coordinate(cv, iqp, coords)
            domainid = inside_domain(x, sd)
            cellquaddomains[iqp, cellid] = domainid
        end
    end
    return cellquaddomains

end



function inside_domain(x::Vec{dim,Float64}, sd::SampleDomain{dim}) where dim

    for inclusion in sd.inclusions
        if point_inside_inclusion(x, inclusion)
            return INCLUSION
        end
    end

    return MATRIX
end

plotdomain(sd::SampleDomain; kwargs...) = plotdomain!(plot(;kwargs...), sd)

function plotdomain!(fig, sd::SampleDomain{2}; kwargs...)

    plot!(fig, [minx(sd.domain), maxx(sd.domain)], [miny(sd.domain), miny(sd.domain)], label = :none, color="black" )
    plot!(fig, [maxx(sd.domain), maxx(sd.domain)], [miny(sd.domain), maxy(sd.domain)], label = :none, color="black" )
    plot!(fig, [minx(sd.domain), maxx(sd.domain)], [maxy(sd.domain), maxy(sd.domain)], label = :none, color="black" )
    plot!(fig, [minx(sd.domain), minx(sd.domain)], [miny(sd.domain), maxy(sd.domain)], label = :none, color="black" )

    for hole in sd.inclusions
        plotinclusion!(fig, hole; kwargs...)
    end

    return fig
end

function plotdomain!(fig, sd::SampleDomain{3}; kwargs...)

    plot3d!(fig, [minx(sd.domain), maxx(sd.domain)], [miny(sd.domain), miny(sd.domain)], [minz(sd.domain), minz(sd.domain)], label = :none, color="red" )
    plot3d!(fig, [maxx(sd.domain), maxx(sd.domain)], [miny(sd.domain), maxy(sd.domain)], [minz(sd.domain), minz(sd.domain)], label = :none, color="red" )
    plot3d!(fig, [minx(sd.domain), maxx(sd.domain)], [maxy(sd.domain), maxy(sd.domain)], [minz(sd.domain), minz(sd.domain)], label = :none, color="red" )
    plot3d!(fig, [minx(sd.domain), minx(sd.domain)], [miny(sd.domain), maxy(sd.domain)], [minz(sd.domain), minz(sd.domain)], label = :none, color="red" )

    for hole in sd.inclusions
        plotinclusion!(fig, hole; kwargs...)
        #scatter3d!(fig, [hole.pos[1]], [hole.pos[2]], [hole.pos[3]]; markershape = :circle, markersize = hole.radius*10, label = :none)
    end

    return fig
end

plotdomain_topview!(fig, sd::SampleDomain{3}) =  _plotdomain_view!(fig, sd, 1, 2)
plotdomain_sideview!(fig, sd::SampleDomain{3}) =  _plotdomain_view!(fig, sd, 1, 3)

function _plotdomain_view!(fig, sd::SampleDomain{3}, d1, d2)

    plot!(fig, [mincoord(sd.domain, dim=d1), maxcoord(sd.domain, dim=d1)], [mincoord(sd.domain, dim=d2), mincoord(sd.domain, dim=d2)], label = :none, color="red" )
    plot!(fig, [maxcoord(sd.domain, dim=d1), maxcoord(sd.domain, dim=d1)], [mincoord(sd.domain, dim=d2), maxcoord(sd.domain, dim=d2)], label = :none, color="red" )
    plot!(fig, [mincoord(sd.domain, dim=d1), maxcoord(sd.domain, dim=d1)], [maxcoord(sd.domain, dim=d2), maxcoord(sd.domain, dim=d2)], label = :none, color="red" )
    plot!(fig, [mincoord(sd.domain, dim=d1), mincoord(sd.domain, dim=d1)], [mincoord(sd.domain, dim=d2), maxcoord(sd.domain, dim=d2)], label = :none, color="red" )

    for hole in sd.inclusions
        plotinclusion!(fig, hole)
    end

    return fig
end

function plotinclusion!(fig, sphere::SphericalInclusion{2}; kwargs...)
    θ = range(0, 2π, 20)
    xc = sphere.pos[1] .+ sphere.radius*cos.(θ)
    yc = sphere.pos[2] .+ sphere.radius*sin.(θ)
    #plot!(fig, xc, yc; kwargs...)
    plot!(fig, xc, yc; label = :none,  fill = (0, 0.7, :gray), w = 0.1, linecolor = :gray)
end

function plotinclusion!(fig, sphere::SphericalInclusion{3}; kwargs...)
    # spherical: (radius r, inclination θ, azimuth φ)
    X(r,theta,phi) = r * sin(theta) * sin(phi)
    Y(r,theta,phi) = r * sin(theta) * cos(phi)
    Z(r,theta,phi) = r * cos(theta)

    thetas = range(0, stop=pi,   length=5)
    phis   = range(0, stop=2pi, length=5)

    r = sphere.radius
    xs = [X(r, theta, phi)+sphere.pos[1] for theta in thetas, phi in phis] 
    ys = [Y(r, theta, phi)+sphere.pos[2] for theta in thetas, phi in phis]
    zs = [Z(r, theta, phi)+sphere.pos[3] for theta in thetas, phi in phis]

    surface!(fig, xs, ys, zs)
end


function plotinclusion!(fig, cyl::CylindricalInclusion; kwargs...)
    @assert abs(cyl.dir ⋅ Vec((1.0,0.0))) ≈ 1.0

    x = cyl.pos[1] - cyl.length/2
    y = cyl.pos[2] - cyl.radius
    w = cyl.length
    h = cyl.radius*2

    xc = [x, x+w, x+w, x, x]
    yc = [y, y, y+h, y+h, y]
    
    plot!(fig, xc, yc; kwargs...)
end

function create_hexahedron(radius::Float64, length::Float64, pos::Vec{dim,Float64}, dir::Vec{dim,Float64}) where dim

    @assert(norm(dir) ≈ 1.0)

    radius *= 1.2
    length *= 1.1

    θ = acos(dir ⋅ Vec((1.0, 0.0, 0.0)))
    xMin = pos[1] - length/2
    xMax = pos[1] + length/2
    yMin = pos[2] - radius
    yMax = pos[2] + radius
    zMin = pos[3] - radius
    zMax = pos[3] + radius

    m = pos

    c = cos(θ)
    s = sin(θ)

    R = [c  -s   0.0; 
         s   c   0.0; 
         0.0 0.0 1.0 ]

    #Rotate and translate
    v1 = SolidModeling.Vertex(R*([xMin, yMin, zMax]-m) + m)
    v2 = SolidModeling.Vertex(R*([xMin, yMax, zMax]-m) + m)
    v3 = SolidModeling.Vertex(R*([xMax, yMax, zMax]-m) + m)
    v4 = SolidModeling.Vertex(R*([xMax, yMin, zMax]-m) + m)
    v5 = SolidModeling.Vertex(R*([xMin, yMin, zMin]-m) + m)
    v6 = SolidModeling.Vertex(R*([xMin, yMax, zMin]-m) + m)
    v7 = SolidModeling.Vertex(R*([xMax, yMax, zMin]-m) + m)
    v8 = SolidModeling.Vertex(R*([xMax, yMin, zMin]-m) + m)

    f1p = SolidModeling.Polygon([v1, v4, v3, v2], SolidModeling.fromPoints(v1.pos, v4.pos, v3.pos))
    f2p = SolidModeling.Polygon([v7, v8, v5, v6], SolidModeling.fromPoints(v7.pos, v8.pos, v5.pos))
    f3p = SolidModeling.Polygon([v1, v2, v6, v5], SolidModeling.fromPoints(v1.pos, v2.pos, v6.pos))
    f4p = SolidModeling.Polygon([v2, v3, v7, v6], SolidModeling.fromPoints(v2.pos, v3.pos, v7.pos))
    f5p = SolidModeling.Polygon([v3, v4, v8, v7], SolidModeling.fromPoints(v3.pos, v4.pos, v8.pos))
    f6p = SolidModeling.Polygon([v4, v1, v5, v8], SolidModeling.fromPoints(v4.pos, v1.pos, v5.pos))

    polys = [f1p, f2p, f3p, f4p, f5p, f6p];

    return SolidModeling.fromPolygons(polys);

end

function test2d()

    dim = 2
    sd = generate_random_domain( AABB( Vec(0.0,0.0), Vec(10.0,10.0)), 0.5, 0.1, 50)
    
    L◫ = 2.0
    subd = cutout_subdomain(sd, L◫)
    
    grid = generate_grid(Quadrilateral, (50,50), subd.domain.corner, subd.domain.corner + subd.domain.lengths)
    ip = Ferrite.default_interpolation(Quadrilateral)
    qr = QuadratureRule{2,RefCube}(2)
    cv = CellScalarValues(qr, ip)

    cellquadpoints = get_qp_domaintags(grid, subd, cv)

    fig = plotdomain(sd)
    plotdomain!(fig, subd)
    display(fig)

    celldomains = [round(Int, mean(col)) for col in eachcol(cellquadpoints)]
    set = collect(1:getncells(grid))[celldomains .== INCLUSION]
    addcellset!(grid, "domains", set)
    vtk_grid("microscale_voxel_$dim", grid) do vtk
        vtk_cellset(vtk, grid)
    end

    #
    grid2 = generate_gmsh(subd)
    vtk_grid("microscale_gmsh_$dim", grid2) do vtk
        vtk_cellset(vtk, grid2)
    end

end


function test3d()

    dim = 3
    domaincoord = Vec{dim,Float64}(d -> 0.0)
    domainsize = Vec{dim,Float64}(d -> 5.0)
    nels = ntuple(d -> d, 30)
    celltype = (dim==2) ? Quadrilateral : Hexahedron

    sd = generate_random_domain( AABB(domaincoord, domainsize), 1.0, 0.1, 20, max_ntries = 10000)
    
    L◫ = 5.0
    subd = cutout_subdomain(sd, L◫)
    
    grid = generate_grid(celltype, (30,30,30), subd.domain.corner, subd.domain.corner + subd.domain.lengths)
    ip = Ferrite.default_interpolation(celltype)
    qr = QuadratureRule{dim,RefCube}(2)
    cv = CellScalarValues(qr, ip)

    cellquadpoints = do_stuff(grid, subd, cv)

    fig = plotdomain(sd)
    #plotdomain!(fig, subd)
    display(fig)

    celldomains = [round(Int, mean(col)) for col in eachcol(cellquadpoints)]
    set = collect(1:getncells(grid))[celldomains .== INCLUSION]
    addcellset!(grid, "domains", set)
    vtk_grid("microscale_voxel_$dim", grid) do vtk
        vtk_cellset(vtk, grid)
    end

    #
    grid2 = generate_gmsh(subd)
    vtk_grid("microscale_gmsh_$dim", grid2) do vtk
        vtk_cellset(vtk, grid2)
    end

end


function test3d_cyl()

    sd = MultiScale.generate_random_domain(MultiScale.CylindricalInclusion{3}, (radius_μ = 0.4, raduis_σ = 0.00001, length_μ=1.0, length_σ = 0.001),  AABB(Vec((0.0, 0.0, 0.0)), Vec((40.0,40.0,40.0))), 100, max_ntries = 20000; allow_outside_domain = true, allow_collisions=false);

    dim = 3
    #L◫ = 2.0
    
    elsize = 1.0
    subd = sd#cutout_inplane_subdomain(sd, L◫)

    #Size
    h = MultiScale.height(subd.domain)
    L◫ = MultiScale.side_length(subd.domain, dim = 1)
    dim==3 && @assert( L◫ == MultiScale.side_length(subd.domain, dim = 2) )

    #Mesh
    nelx = round(Int, L◫/elsize)
    nelz = round(Int, h/elsize)
    nels =  ntuple(d-> d==dim ? nelz : nelx, dim) 
    corner = Vec{dim,Float64}( d-> d == dim ? h/2 : L◫/2 )
    celltype = dim == 2 ? Quadrilateral : Hexahedron
    grid = generate_grid(celltype, nels, -corner, corner) 

    #Cellvalues
    ip = Ferrite.default_interpolation(celltype)
    shape = Ferrite.getrefshape(ip)
    qr = QuadratureRule{dim,shape}(2)
    cv = CellScalarValues(qr, ip)
    
    #Get the material in each qp in each element
    celldomains = get_qp_domaintags(grid, subd, cv)

    #Add inclusions cellset for vtk visulations
    _celldomains = [maximum(col) for col in eachcol(celldomains)]
    grid.cellsets["inclusions"] = findall(i -> i == MultiScale.INCLUSION, _celldomains) |> Set

    vtk_grid("mygridsave", grid) do vtk
        vtk_cellset(vtk, grid)
    end
end