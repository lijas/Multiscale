using Random, Distributions, Tensors
using Plots, Ferrite
plotly()

const INCLUSION=1 
const MATRIX=2

struct AABB{dim}
    corner::Vec{dim, Float64}
    lengths::Vec{dim, Float64}
end

function AABB(c::Vec{dim,Float64}, l::Vec{dim,Float64}) where dim
    return AABB{dim}(c,l)
end

minx(a::AABB) = a.corner[1]
miny(a::AABB) = a.corner[2]
maxx(a::AABB) = a.corner[1] + a.lengths[1]
maxy(a::AABB) = a.corner[2] + a.lengths[2]

mincoord(a::AABB; dim::Int) = a.corner[dim]
maxcoord(a::AABB; dim::Int) = a.corner[dim] + a.lengths[dim]

struct GradedElasticMaterial{N}
    Es::NTuple{N,Float64}
    νs::NTuple{N,Float64}
end

struct GradedElasticMaterialState{N}
    domaintag::NTuple{N,Int}
end

struct Inclusion{dim}
    radius::Float64
    pos::Vec{dim,Float64}
end

struct SampleDomain{dim}
    inclusions::Vector{Inclusion{dim}}
    domain::AABB{dim}
end

SampleDomain(aabb::AABB{dim}) where dim = SampleDomain{dim}(Inclusion{dim}[], aabb)


include("gmshdomain.jl")

function generate_random_domain(aabb::AABB{dim}, radius_μ, radius_σ, ninclusions::Int; max_ntries::Int = ninclusions*10) where dim

    inclusions = Inclusion{dim}[]
    
    ndradius = Normal(radius_μ, radius_σ)
    udpos = ntuple(d -> Uniform( mincoord(aabb, dim=d), maxcoord(aabb, dim=d)), dim)
    
    nadded = 0
    ntries = 0
    while nadded < ninclusions
        pos_test = generate_random_position(udpos)
        radius_test = generate_random_radius(ndradius)
        
        new_inclusion = Inclusion(radius_test, pos_test)

        if collides_with_other_inclusion(inclusions, new_inclusion)
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

    @show (nadded)
    @show (ntries)
    @show length(inclusions)
    return SampleDomain(inclusions, aabb)
end

generate_random_position(uniform::NTuple{dim, Uniform}) where dim = Vec{dim,Float64}( (d) -> rand(uniform[d]) )
generate_random_radius(normal::Normal) where dim = rand(normal)


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

function cutout_subdomain(sd::SampleDomain{dim}, L◫::Float64) where dim
    udpos = ntuple(d -> Uniform( mincoord(sd.domain, dim=d), maxcoord(sd.domain, dim=d)), dim)

    h = L◫
    size = Vec{dim,Float64}( d -> d==3 ? h : L◫ )
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



function do_stuff(grid::Grid, sd::SampleDomain, cv::Ferrite.Values)

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

plotdomain(sd::SampleDomain) = plotdomain!(plot(reuse=false, legend=false, axis_equal=true), sd)

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

function plotcircle!(fig, (x,y)::Vec{2,Float64},r::Float64; kwargs...)
    θ = range(0, 2π, 20)
    xc = x .+ r*sin.(θ)
    yc = y .+ r*cos.(θ)
    plot!(fig, xc, yc; kwargs...)
end


function test()

    dim = 2
    sd = generate_random_domain( AABB( Vec(0.0,0.0), Vec(10.0,10.0)), 0.5, 0.1, 60)
    
    L◫ = 2.0
    subd = cutout_subdomain(sd, L◫)
    
    grid = generate_grid(Quadrilateral, (50,50), subd.domain.corner, subd.domain.corner + subd.domain.lengths)
    ip = Ferrite.default_interpolation(Quadrilateral)
    qr = QuadratureRule{2,RefCube}(2)
    cv = CellScalarValues(qr, ip)

    cellquadpoints = do_stuff(grid, subd, cv)

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

    sd = generate_random_domain( AABB(domaincoord, domainsize), 0.5, 0.1, 100)
    
    L◫ = 5.0
    subd = cutout_subdomain(sd, L◫)
    
    grid = generate_grid(celltype, (30,30,30), subd.domain.corner, subd.domain.corner + subd.domain.lengths)
    ip = Ferrite.default_interpolation(celltype)
    qr = QuadratureRule{dim,RefCube}(2)
    cv = CellScalarValues(qr, ip)

    cellquadpoints = do_stuff(grid, subd, cv)

    #fig = plotdomain(sd)
    #plotdomain!(fig, subd)
    #display(fig)

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
