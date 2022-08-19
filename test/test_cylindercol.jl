

@testset "cylinder col" begin

    #CylindricalInclusion(radius::Float64, length::Float64, pos::Vec{dim,Float64}, dir::Vec{dim,Float64})
    a = MultiScale.CylindricalInclusion(3.0, 10.0, Vec((0.0,0.0,0.0)), Vec((1.0,0.0,0.0)))
    b = MultiScale.CylindricalInclusion(2.0, 14.0, Vec((1.0,0.0,0.0)), Vec((0.0,1.0,0.0)))

    @show MultiScale._collides(a,b)  

    #parallel
    b = MultiScale.CylindricalInclusion(3.0, 10.0, Vec((2.0,0.0,2.0)), Vec((1.0,0.0,0.0)))

    @show MultiScale.point_inside_inclusion(Vec((1.0,0.0,0.0)), a)
    @show MultiScale.point_inside_inclusion(Vec((5.01,0.0,0.0)), a)
    @show MultiScale.point_inside_inclusion(Vec((-5.01,0.0,0.0)), a)
    @show MultiScale.point_inside_inclusion(Vec((4.99,0.0,0.0)), a)
    @show MultiScale.point_inside_inclusion(Vec((-4.99,0.0,0.0)), a)
    @show MultiScale.point_inside_inclusion(Vec((1.0, 1.0, 1.0)), a)
    @show MultiScale.point_inside_inclusion(Vec((1.0, 0.0, 2.999)), a) == true
    @show MultiScale.point_inside_inclusion(Vec((1.0, 0.0, 3.001)), a) == false

    #
    #Two parallel collides
    #
    dir = Vec((1.0, 0.0, 0.0))
    pos = Vec((2.0, 2.0, 2.0))
    a = MultiScale.CylindricalInclusion(2.0, 10.0, pos + Vec((0.0, 0.0, 0.0)), dir)
    b = MultiScale.CylindricalInclusion(2.0, 10.0, pos + Vec((1.5, 0.0, 2.0)), dir)
    @test MultiScale._collides(a,b) == true


    dir = Vec((rand(), rand(), 0.0))
    dir /= norm(dir)
    pos = Vec((2.0, 2.0, 2.0))
    a = MultiScale.CylindricalInclusion(2.0, 10.0, pos + Vec((0.0, 0.0, 0.0)), dir)
    b = MultiScale.CylindricalInclusion(2.0, 10.0, pos + Vec((1.5, 0.0, 2.0)), dir)
    @test MultiScale._collides(a,b) == true

    #Radius not overlapping
    dir = Vec((rand(), rand(), 0.0))
    dir /= norm(dir)
    pos = Vec((2.0, 2.0, 2.0))
    a = MultiScale.CylindricalInclusion(2.0, 10.0, pos + Vec((0.0, 0.0, 0.0)), dir)
    b = MultiScale.CylindricalInclusion(2.0, 10.0, pos + Vec((1.5, 0.0, 5.0)), dir)
    @test MultiScale._collides(a,b) == false

    #Direction not overlapping
    dir = Vec((rand(), rand(), 0.0))
    dir /= norm(dir)
    pos = Vec((2.0, 2.0, 2.0))
    a = MultiScale.CylindricalInclusion(2.0, 10.0, pos + Vec((0.0, 0.0, 0.0)), dir)
    b = MultiScale.CylindricalInclusion(2.0, 10.0, pos + Vec((40., 0.0, 2.0)), dir)
    @test MultiScale._collides(a,b) == false
end