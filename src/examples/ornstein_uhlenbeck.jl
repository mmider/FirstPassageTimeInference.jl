using StaticArrays, LinearAlgebra
using GaussianDistributions

struct OrnsteinUhlenbeck <: ContinuousTimeProcess{Float64}
    θ::Float64
    ρ::Float64
    σ::Float64
end


current(t, ::OrnsteinUhlenbeck) = 0.0
current_prime(t, ::OrnsteinUhlenbeck) = 0.0
state_space(::OrnsteinUhlenbeck) = Unrestricted()

b(t, y::Float64, P::OrnsteinUhlenbeck) = P.ρ - P.θ*y + current(t, P)
σ(t, y::Float64, P::OrnsteinUhlenbeck) = P.σ

function ϕ(t, y::Float64, P::OrnsteinUhlenbeck)
    0.5*((b(t,P.σ*y,P)/P.σ)^2-P.θ) + current_prime(t, P)/P.σ*y
end

function A(t, y::Float64, P::OrnsteinUhlenbeck)
    (P.ρ+current(t, P))/P.σ*y - 0.5*P.θ*y^2
end

η(t, y::Float64, P::OrnsteinUhlenbeck) = y/P.σ

η⁻¹(y::Float64, P::OrnsteinUhlenbeck) = η⁻¹(nothing, y, P)
η⁻¹(t, y::Float64, P::OrnsteinUhlenbeck) = P.σ*y

params(P::OrnsteinUhlenbeck) = [P.θ, P.ρ, P.σ]

clone(P::OrnsteinUhlenbeck, θ) = OrnsteinUhlenbeck(θ...)

φ(t, y::Float64, P::OrnsteinUhlenbeck) = @SVector[-y, 1.0/P.σ]
φᶜ(t, y::Float64, P::OrnsteinUhlenbeck) = current(t, P)/P.σ
