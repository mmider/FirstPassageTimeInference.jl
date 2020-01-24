struct LangevinT <: ContinuousTimeProcess{Float64}
    v::Float64
    μ::Float64
    σ::Float64
end

current(t, ::LangevinT) = 0.0
current_prime(t, ::LangevinT) = 0.0
state_space(::LangevinT) = Unrestricted()

function b(t, y::Float64, P::LangevinT)
    x = y - P.μ
    -0.5*(P.v + 1.0)*x/(P.v + x^2) + current(t, P)
end

σ(t, y::Float64, P::LangevinT) = P.σ

function α(t, y::Float64, P::LangevinT)
    x = y - P.μ/P.σ
    -0.5*(P.v + 1.0)*x/(P.v + (P.σ*x)^2) + current(t, P)/P.σ
end

function α_prime(t, y::Float64, P::LangevinT)
    x = P.σ*y - P.μ
    0.5*(P.v + 1.0) * (x^2 - P.v)/(P.v + x^2)^2
end

function ϕ(t, y::Float64, P::LangevinT)
    ∂ₜA = y*current_prime(t, P)/P.σ
    return 0.5*(α(t,y,P)^2 + α_prime(t,y,P)) + ∂ₜA
end

function A(t, y::Float64, P::LangevinT)
    x = y - P.μ/P.σ
    -0.25*(P.v+1.0)*log(P.v+(x*P.σ)^2)/P.σ^2 + y*current(t, P)/P.σ
end

η(t, y::Float64, P::LangevinT) = y/P.σ

η⁻¹(y::Float64, P::LangevinT) = η⁻¹(nothing, y, P)
η⁻¹(t, y::Float64, P::LangevinT) = P.σ*y

params(P::LangevinT) = [P.v, P.μ, P.σ]

clone(P::LangevinT, θ) = LangevinT(θ...)
