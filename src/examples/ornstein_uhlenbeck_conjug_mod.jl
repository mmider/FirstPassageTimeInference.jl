using StaticArrays, LinearAlgebra
using GaussianDistributions

struct OrnsteinUhlenbeckMod{T,S} <: ContinuousTimeProcess{Float64}
    θ::Float64
    ρ::Float64
    σ::Float64
    f::T
    f_prime::S
    function OrnsteinUhlenbeckMod(θ, ρ, σ, f::T, f_prime::S) where {T,S}
        new{T,S}(θ, ρ, σ, f, f_prime)
    end
end

state_space(::OrnsteinUhlenbeckMod) = Unrestricted()

b(t, y::Float64, P::OrnsteinUhlenbeckMod) = P.ρ-P.θ*y+P.f(t)
σ(t, y::Float64, P::OrnsteinUhlenbeckMod) = P.σ

function ϕ(t, y::Float64, P::OrnsteinUhlenbeckMod)
    0.5*((b(t,P.σ*y,P)/P.σ)^2-P.θ) + P.f_prime(t)/P.σ*y
end

function A(t, y::Float64, P::OrnsteinUhlenbeckMod)
    (P.ρ+P.f(t))/P.σ*y - 0.5*P.θ*y^2
end

η(t, y::Float64, P::OrnsteinUhlenbeckMod) = y/P.σ

η⁻¹(t, y::Float64, P::OrnsteinUhlenbeckMod) = P.σ*y

params(P::OrnsteinUhlenbeckMod) = [P.θ, P.ρ, P.σ]

clone(P::OrnsteinUhlenbeckMod, θ) = OrnsteinUhlenbeckMod(θ..., P.f, P.f_prime)

φ(t, y::Float64, P::OrnsteinUhlenbeckMod) = @SVector[-y, 1.0/P.σ]
φᶜ(t, y::Float64, P::OrnsteinUhlenbeckMod) = P.f(t)/P.σ

function conjugateDraw(θ, XX, P::OrnsteinUhlenbeckMod, prior)
    μ = @SVector[0.0, 0.0]
    𝓦 = μ*μ'
    ϑ = @SVector[θ[1], θ[2]]
    μ, 𝓦 = _conjugateDraw(μ, 𝓦, XX, P)

    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2
    μₚₒₛₜ = Σ * (μ + Vector(prior.Σ\prior.μ))

    ϑᵒ = rand(Gaussian(μₚₒₛₜ, Σ))
    θᵒ = copy(θ)
    θᵒ[1], θᵒ[2] = ϑᵒ[1], ϑᵒ[2]
    θᵒ
end

function _conjugateDraw(μ, 𝓦, XX, P::OrnsteinUhlenbeckMod)
    for X in XX
        for i in 1:length(X)-1
            φₜ = φ(X.tt[i], X.yy[i], P)
            φₜᶜ = φᶜ(X.tt[i], X.yy[i], P)
            dt = X.tt[i+1]-X.tt[i]
            dy = X.yy[i+1]-X.yy[i]
            μ = μ + φₜ*dy - φₜ*φₜᶜ*dt
            𝓦 = 𝓦 + φₜ*φₜ'*dt
        end
    end
    μ, 𝓦
end
