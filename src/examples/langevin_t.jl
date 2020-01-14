struct LangevinT <: ContinuousTimeProcess{Float64}
    v::Float64
    μ::Float64
    σ::Float64
end

current(t, ::LangevinT) = 0.0
current_prime(t, ::LangevinT) = 0.0
state_space(::LangevinT) = Unrestricted()

function b(t, y::Float64, P::LangevinT)
    x = y - P.μ + current(t, P)
    -0.5*(P.v + 1.0)*x/(P.v + x^2)
end
σ(t, y::Float64, P::LangevinT) = P.σ


function α(t, y::Float64, P::LangevinT)
    x = P.σ*y - P.μ + current(t, P)
    -0.5*(P.v + 1.0)*x/(P.σ*(P.v + x^2))
end

function α_prime(t, y::Float64, P::LangevinT)
    x = P.σ*y - P.μ + current(t, P)
    0.5*(P.v + 1.0) * (x^2 - P.v)/(P.v + x^2)^2
end

function ϕ(t, y::Float64, P::LangevinT)
    return 0.5*(α(t,y,P)^2 + α_prime(t,y,P)) + α(t,y,P)*current_prime(t, P)/P.σ
end

function A(t, y::Float64, P::LangevinT)
    x = P.σ*y - P.μ + current(t, P)
    -0.25*(P.v+1.0)*log(1+x^2/P.v)/P.σ^2
end

η(t, y::Float64, P::LangevinT) = y/P.σ

η⁻¹(y::Float64, P::LangevinT) = η⁻¹(nothing, y, P)
η⁻¹(t, y::Float64, P::LangevinT) = P.σ*y

params(P::LangevinT) = [P.v, P.μ, P.σ]

clone(P::LangevinT, θ) = LangevinT(θ...)
