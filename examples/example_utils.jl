function calculate_anlytical(material, macroscale::MacroParameters, angles = [0.0], coords = [-0.5, 0.5])

    state = MaterialModels.initial_material_state(material)
    _, E, _ = material_response(PlaneStress(), material, rand(SymmetricTensor{2,2}), state);
    QLT = [E[1,1,1,1] E[1,1,2,2] 0.0;
           E[2,2,1,1] E[2,2,2,2] 0.0;
           0.0 0.0 E[1,2,1,2]]

    _, E, _ = material_response(material, rand(SymmetricTensor{2,3}), state);
    QLT′ = [E[1,2,1,2] 0.0;
            0.0 E[2,3,2,3]]

    A, B, D, Ã = ABDMatrix(QLT, QLT′, angles, coords);

    A = fromvoigt(SymmetricTensor{4,2}, A)
    B = fromvoigt(SymmetricTensor{4,2}, B)
    D = fromvoigt(SymmetricTensor{4,2}, D)
    Ã = SymmetricTensor{2,2}((i,j)->Ã[i,j]) 

    N = A⊡symmetric(macroscale.∇u) - B⊡macroscale.∇θ
    M = B⊡symmetric(macroscale.∇u) - D⊡macroscale.∇θ
    V = Ã ⋅ (macroscale.∇w - macroscale.θ)
    
    return N, M, V
end

function ABDMatrix(QLT, QLT′, angles::Vector{Float64}, coord::AbstractVector{Float64})
    # Function calculating A, B and D matrix for a laminate

    # Inputs:
    #        QLT: Local stiffness matrix
    #        ang: vector with the angles of the lamina (starting from min. z)
    #        coord: vector with z-coordinates of the lamina interfaces
    #        alfaLT: Local heat expansion vector
    #        deltaT: Difference in temperature

    T1(Θ)= [cos(Θ)^2 sin(Θ)^2  2*sin(Θ)*cos(Θ);
            sin(Θ)^2 cos(Θ)^2 -2*sin(Θ)*cos(Θ);
            -sin(Θ)*cos(Θ) sin(Θ)*cos(Θ) (cos(Θ)^2-sin(Θ)^2)]

    T2(Θ) =  [cos(Θ)^2 sin(Θ)^2  sin(Θ)*cos(Θ);
                sin(Θ)^2 cos(Θ)^2 -sin(Θ)*cos(Θ);
                -2*sin(Θ)*cos(Θ) 2*sin(Θ)*cos(Θ) (cos(Θ)^2-sin(Θ)^2)]

    
    
    T1′(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)];

    A = zeros(Float64, 3, 3) 
    B = zeros(Float64, 3, 3) 
    D = zeros(Float64, 3, 3)
    Ã = zeros(Float64, 2, 2);

    Qxy(Θ) = inv(T1(Θ))*QLT*T2(Θ)
    Qxy′(θ) = inv(T1′(θ))*QLT′*T1′(θ);
    
    nla = length(angles)

    for ii = 1:nla
        Θ = angles[ii]
        A += 1/1*Qxy(Θ)*(coord[ii+1]^1-coord[ii]^1);
        B += 1/2*Qxy(Θ)*(coord[ii+1]^2-coord[ii]^2);
        D += 1/3*Qxy(Θ)*(coord[ii+1]^3-coord[ii]^3);
        Ã += 1/1*Qxy′(Θ)*(coord[ii+1]^1-coord[ii]^1);
    end

    return A, B, D, Ã
end