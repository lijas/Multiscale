export AABB, SampleDomain, Inclusion
export generate_random_domain, cutout_inplane_subdomain, get_qp_domaintags
export plotdomain, plotdomain!, plotdomain_topview!, plotdomain_sideview!

#using Plots; plotly()

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

mincoord(a::AABB; dim::Int) = a.corner[dim]
maxcoord(a::AABB; dim::Int) = a.corner[dim] + a.lengths[dim]
struct Inclusion{dim}
    radius::Float64
    pos::Vec{dim,Float64}
    function Inclusion(radius::Float64, pos::Vec{dim,Float64}) where dim
        @assert radius > 0.0
        return new{dim}(radius, pos)
    end
end

struct SampleDomain{dim}
    inclusions::Vector{Inclusion{dim}}
    domain::AABB{dim}
end

function offset_domain(sd::SampleDomain, offset)

    new_spheres = similar(sd.inclusions)

    for i in 1:length(sd.inclusions) 
        r = sd.inclusions[i].radius
        pos = sd.inclusions[i].pos - offset
        new_spheres[i] = Inclusion(r, pos)
    end

    return SampleDomain( new_spheres, offset_aabb(sd.domain, offset))
end


SampleDomain(aabb::AABB{dim}) where dim = SampleDomain{dim}(Inclusion{dim}[], aabb)

function generate_random_domain(aabb::AABB{dim}, radius_μ, radius_σ, ninclusions::Int; max_ntries::Int = ninclusions*10) where dim
    @assert( radius_μ - radius_σ > 0.0)

    inclusions = Inclusion{dim}[]
    
    ndradius = Uniform(radius_μ - radius_σ, radius_μ + radius_σ) #Normal(radius_μ, radius_σ)
    udpos = ntuple(d -> Uniform( mincoord(aabb, dim=d), maxcoord(aabb, dim=d)), dim)
    
    nadded = 0
    ntries = 0
    while nadded < ninclusions
        pos_test = generate_random_position(udpos)
        radius_test = generate_random_radius(ndradius)
        new_inclusion = Inclusion(radius_test, pos_test)

        if collides_with_other_inclusion(inclusions, new_inclusion) || inclusion_outside_domain(aabb, new_inclusion)
            if ntries > max_ntries
                error("Could not distribute all voids: $nadded/$ninclusions")
            end
            ntries += 1
            continue
        else
            nadded += 1
            push!(inclusions, new_inclusion)
        end
        

    end

    return SampleDomain(inclusions, aabb)
end

generate_random_position(uniform::NTuple{dim, Uniform}) where dim = Vec{dim,Float64}( (d) -> rand(uniform[d]) )
generate_random_radius(normal::ContinuousUnivariateDistribution)= rand(normal)


function collides_with_other_inclusion(inclusions::Vector{Inclusion{dim}}, new_inclusion::Inclusion) where dim

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

function inclusion_outside_domain(aabb::AABB{dim}, a::Inclusion{dim}) where dim

    for d in 1:dim
        if (a.pos[d] - a.radius) < mincoord(aabb, dim = d)
            return true
        elseif (a.pos[d] + a.radius) > maxcoord(aabb, dim = d)
            return true
        end
    end
    return false
end

function cutout_subdomain(sd::SampleDomain{dim}, L◫::Float64) where dim
    udpos = ntuple(d -> Uniform( mincoord(sd.domain, dim=d), maxcoord(sd.domain, dim=d)), dim)

    h = L◫
    size = Vec{dim,Float64}( d -> d==dim ? h : L◫ )
    x = generate_random_position(udpos)
   
    sub_aabb = AABB(x - 0.5*size , size)
    subdomain = SampleDomain(sub_aabb)

    for sphere in sd.inclusions
        if sphere_inside_aabb(sphere, sub_aabb)
            push!(subdomain.inclusions, sphere)
        end
    end
    @show length(subdomain.inclusions)
    return subdomain
end

function cutout_inplane_subdomain(sd::SampleDomain{dim}, L◫::Float64) where dim
    h = height(sd.domain)
    udpos = ntuple(d -> Uniform( mincoord(sd.domain, dim=d), maxcoord(sd.domain, dim=d)), dim-1)

    size  = Vec{dim}( d -> d==dim ? h : L◫ )
    coord = Vec{dim}( d -> d!=dim ? rand(udpos[d]) : mincoord(sd.domain; dim=d) )
   
    sub_aabb = AABB(coord , size)
    subdomain = SampleDomain(sub_aabb)

    for sphere in sd.inclusions
        if sphere_inside_aabb(sphere, sub_aabb)
            push!(subdomain.inclusions, sphere)
        end
    end
    @show length(subdomain.inclusions)
    return subdomain
end

function sphere_inside_aabb(sphere::Inclusion{dim}, aabb::AABB{dim}) where dim

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

    for sphere in sd.inclusions
        if norm(sphere.pos - x) ≤ sphere.radius
            return INCLUSION
        end
    end

    return MATRIX
end

plotdomain(sd::SampleDomain; kwargs...) = plotdomain!(plot(;kwargs...), sd)

function plotdomain!(fig, sd::SampleDomain{2})

    plot!(fig, [minx(sd.domain), maxx(sd.domain)], [miny(sd.domain), miny(sd.domain)], color="red" )
    plot!(fig, [maxx(sd.domain), maxx(sd.domain)], [miny(sd.domain), maxy(sd.domain)], color="red" )
    plot!(fig, [minx(sd.domain), maxx(sd.domain)], [maxy(sd.domain), maxy(sd.domain)], color="red" )
    plot!(fig, [minx(sd.domain), minx(sd.domain)], [miny(sd.domain), maxy(sd.domain)], color="red" )

    for hole in sd.inclusions
        plotcircle!(fig, hole.pos, hole.radius)
    end

    return fig
end

function plotdomain!(fig, sd::SampleDomain{3})

    plot3d!(fig, [minx(sd.domain), maxx(sd.domain)], [miny(sd.domain), miny(sd.domain)], [minz(sd.domain), minz(sd.domain)], color="red" )
    plot3d!(fig, [maxx(sd.domain), maxx(sd.domain)], [miny(sd.domain), maxy(sd.domain)], [minz(sd.domain), minz(sd.domain)], color="red" )
    plot3d!(fig, [minx(sd.domain), maxx(sd.domain)], [maxy(sd.domain), maxy(sd.domain)], [minz(sd.domain), minz(sd.domain)], color="red" )
    plot3d!(fig, [minx(sd.domain), minx(sd.domain)], [miny(sd.domain), maxy(sd.domain)], [minz(sd.domain), minz(sd.domain)], color="red" )

    for hole in sd.inclusions
        scatter3d!(fig, [hole.pos[1]], [hole.pos[2]], [hole.pos[3]]; markershape = :circle, markersize = hole.radius*10)
    end

    return fig
end

plotdomain_topview!(fig, sd::SampleDomain{3}) =  _plotdomain_view!(fig, sd, 1, 2)
plotdomain_sideview!(fig, sd::SampleDomain{3}) =  _plotdomain_view!(fig, sd, 1, 3)

function _plotdomain_view!(fig, sd::SampleDomain{3}, d1, d2)

    plot!(fig, [mincoord(sd.domain, dim=d1), maxcoord(sd.domain, dim=d1)], [mincoord(sd.domain, dim=d2), mincoord(sd.domain, dim=d2)], color="red" )
    plot!(fig, [maxcoord(sd.domain, dim=d1), maxcoord(sd.domain, dim=d1)], [mincoord(sd.domain, dim=d2), maxcoord(sd.domain, dim=d2)], color="red" )
    plot!(fig, [mincoord(sd.domain, dim=d1), maxcoord(sd.domain, dim=d1)], [maxcoord(sd.domain, dim=d2), maxcoord(sd.domain, dim=d2)], color="red" )
    plot!(fig, [mincoord(sd.domain, dim=d1), mincoord(sd.domain, dim=d1)], [mincoord(sd.domain, dim=d2), maxcoord(sd.domain, dim=d2)], color="red" )

    for hole in sd.inclusions
        plotcircle!(fig, Vec((hole.pos[d1], hole.pos[d2])), hole.radius)
    end

    return fig
end

function plotcircle!(fig, (x,y)::Vec{2,Float64}, r::Float64; kwargs...)
    θ = range(0, 2π, 20)
    xc = x .+ r*cos.(θ)
    yc = y .+ r*sin.(θ)
    plot!(fig, xc, yc; kwargs...)
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


