
Base.Base.@kwdef struct TransverseIsotropy{T} <: MaterialModels.AbstractMaterial
    L⊥::T
    L₌::T
    M₌::T
    G⊥::T
    G₌::T
end

struct TransverseIsotropyState{T} <: MaterialModels.AbstractMaterialState
    σ::SymmetricTensor{2,3,T,6}
    a::Vec{3,Float64}
end

function MaterialModels.initial_material_state(m::TransverseIsotropy)
    return TransverseIsotropyState(zero(SymmetricTensor{2,3,Float64,6}), zero(Vec{3,Float64}))
end

function MaterialModels.initial_material_state(m::TransverseIsotropy, a::Vec{3,Float64})
    return TransverseIsotropyState(zero(SymmetricTensor{2,3,Float64,6}), a)
end

function MaterialModels.material_response(m::TransverseIsotropy, ε::SymmetricTensor{2,3}, state::TransverseIsotropyState, Δt=nothing; cache=nothing, options=nothing)

    a3 = state.a
    I = one(ε)
    A = symmetric( a3 ⊗ a3 )
    𝔸 = 0.25 * symmetric( otimesu(A,I) + otimesl(A,I) + otimesu(I,A) + otimesl(I,A) )
    Iˢʸᵐ = 0.5 * symmetric( (otimesu(I,I) + otimes(I,I)) )
    
    #@assert( ismajorsymmetric(𝔸) )
    #@assert( isminorsymmetric(𝔸) )

    E = m.L⊥                                  * symmetric(I ⊗ I)     + 
        2m.G⊥                                 * Iˢʸᵐ                 + 
        (m.L₌ - m.L⊥)                         * symmetric(I⊗A + A⊗I) + 
        (m.M₌ - 4m.G₌ + 2m.G⊥ - 2m.L₌ + m.L⊥) * symmetric(A ⊗ A)     + 
        4(m.G₌ - m.G⊥)                        * 𝔸

    #@assert( ismajorsymmetric(E) )
    #@assert( isminorsymmetric(E) )
    σ = E ⊡ ε

    return σ, E, TransverseIsotropyState(σ, state.a)

end