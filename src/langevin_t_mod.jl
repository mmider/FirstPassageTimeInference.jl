struct LangevinTMod{T,S} <: ContinuousTimeProcess{Float64}
    v::Float64
    μ::Float64
    σ::Float64
    f::T
    f_prime::S

    function LangevinTMod(v, μ, σ, f::T, f_prime::S) where {T,S}
        new{T,S}(v, μ, σ, f, f_prime)
    end
end

function b(t, y::Float64, P::LangevinTMod)
    x = y-P.μ+P.f(t)
    -0.5*(P.v+1.0)*x/(P.v+x^2)
end
σ(t, y::Float64, P::LangevinTMod) = P.σ


function α(t, y::Float64, P::LangevinTMod)
    x = P.σ*y-P.μ+P.f(t)
    -0.5*(P.v+1.0)*x/(P.σ*(P.v+x^2))
end

function α_prime(t, y::Float64, P::LangevinTMod)
    x = P.σ*y-P.μ+P.f(t)
    0.5*(P.v+1.0)*(x^2-P.v)/(P.v+x^2)^2
end

function ϕ(t, y::Float64, P::LangevinTMod)
    return 0.5*(α(t,y,P)^2 + α_prime(t,y,P)) + α(t,y,P)*P.f_prime(t)/P.σ
end

function A(t, y::Float64, P::LangevinTMod)
    x = P.σ*y-P.μ+P.f(t)
    -0.25*(P.v+1.0)*log(1+x^2/P.v)/P.σ^2
end

η(t, y::Float64, P::LangevinTMod) = y/P.σ

η⁻¹(t, y::Float64, P::LangevinTMod) = P.σ*y

params(P::LangevinTMod) = [P.v, P.μ, P.σ]

clone(P::LangevinTMod, θ) = LangevinTMod(θ..., P.f, P.f_prime)
