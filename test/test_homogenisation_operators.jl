



@testset "Test homogenisation operators" begin

    mp = MultiScale.MacroParameters{2}(
        ∇u = Tensor{2,2}((0.1, 0.4, 0.5, 0.6)), 
        ∇w = Vec{2}((0.7, 0.8)), 
        ∇θ = Tensor{2,2}((1.9, 0.3, 0.4, 0.5)), 
        w = 3.0, 
        θ = Vec{2}((0.4,0.5)),
        u = Vec{2}((0.4,0.5))
    )
    EXTRA_PROLONGATION = false
    x̄ = Vec((1.0,2.0,0.0))#Vec((1.0,2.0,3.0))
    size = Vec((2.0,5.0,6.0))
    Ω = prod(size)
    I = (size[1]*size[2]*size[3]^3)/12

    grid = Ferrite.generate_grid(Hexahedron, (5,5,5), x̄-size/2, x̄+size/2)
    dh = DofHandler(grid)
    push!(dh, :u, 3)
    close!(dh)
    
    uM = zeros(Float64, ndofs(dh))
    chM = ConstraintHandler(dh)
    add!(chM, Ferrite.Dirichlet(:u, Set(1:getnnodes(grid)), (x, t) -> MultiScale.prolongation(x, x̄, mp, EXTRA_PROLONGATION), 1:3))
    close!(chM)
    update!(chM, 0.0)
    apply!(uM, chM)

    ip = Ferrite.default_interpolation(getcelltype(grid))
    cv = CellVectorValues(Ferrite._mass_qr(ip), ip)

    u◫ = zero(Vec{3})
    w◫ = zero(Float64)
    θ◫ = zero(Vec{3})
    h◫ = zero(Tensor{2,3})
    g◫ = zero(Vec{3})
    κ◫ = zero(Tensor{2,3})
    for cell in CellIterator(dh)
        x = getcoordinates(cell)
        ae = uM[celldofs(cell)]
        reinit!(cv, x)
        u◫ += 1/Ω * MultiScale.u◫_operator(cv, ae)
        w◫ += 1/Ω * MultiScale.w◫_operator(cv, ae)
        θ◫ -= 1/I * MultiScale.θ◫_operator(cv, ae, x)
        h◫ += 1/Ω * MultiScale.h◫_operator(cv, ae)
        g◫ += 1/Ω * MultiScale.g◫_operator(cv, ae)
        κ◫ -= 1/I * MultiScale.κ◫_operator(cv, ae, x)
    end

    @test u◫ ≈ MultiScale.increase_dim(mp.u)
    @test w◫ ≈ mp.w
    @test θ◫ ≈ MultiScale.increase_dim(mp.θ)
    @test h◫ ≈ MultiScale.increase_dim(mp.∇u)
    @test g◫ ≈ MultiScale.increase_dim(mp.∇w)
    @test κ◫ ≈ MultiScale.increase_dim(mp.∇θ)
end