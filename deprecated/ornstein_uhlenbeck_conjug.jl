using GaussianDistributions

struct OrnsteinUhlenbeckAlt <: ContinuousTimeProcess{Float64}
    θ::Float64
    ρ::Float64
    σ::Float64
end

state_space(::OrnsteinUhlenbeckAlt) = Unrestricted()

b(t, y::Float64, P::OrnsteinUhlenbeckAlt) = P.ρ-P.θ*y
σ(t, y::Float64, P::OrnsteinUhlenbeckAlt) = P.σ

ϕ(y::Float64, P::OrnsteinUhlenbeckAlt) = ϕ(nothing, y, P)
ϕ(t, y::Float64, P::OrnsteinUhlenbeckAlt) = 0.5*((P.ρ/P.σ-P.θ*y)^2-P.θ)

A(y::Float64, P::OrnsteinUhlenbeckAlt) = A(nothing, y, P)
A(t, y::Float64, P::OrnsteinUhlenbeckAlt) = P.ρ/P.σ*y - 0.5*P.θ*y^2

η(y::Float64, P::OrnsteinUhlenbeckAlt) = η(nothing, y, P)
η(t, y::Float64, P::OrnsteinUhlenbeckAlt) = y/P.σ

η⁻¹(y::Float64, P::OrnsteinUhlenbeckAlt) = η⁻¹(nothing, y, P)
η⁻¹(t, y::Float64, P::OrnsteinUhlenbeckAlt) = P.σ*y

params(P::OrnsteinUhlenbeckAlt) = [P.θ, P.ρ, P.σ]

clone(P::OrnsteinUhlenbeckAlt, θ) = OrnsteinUhlenbeckAlt(θ...)

φ(y::Float64, P::OrnsteinUhlenbeckAlt) = φ(nothing, y, P)
φ(t, y::Float64, P::OrnsteinUhlenbeckAlt) = @SVector[-y, 1.0/P.σ]

function conjugateDraw(θ, XX, P::T, prior) where T <: Union{OrnsteinUhlenbeckAlt,OrnsteinUhlenbeckMod,CoxIngersollRoss}
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

function _conjugateDraw(μ, 𝓦, XX, P::OrnsteinUhlenbeckAlt)
    for X in XX
        for i in 1:length(X)-1
            φₜ = φ(X.yy[i], P)
            dt = X.tt[i+1]-X.tt[i]
            dy = X.yy[i+1]-X.yy[i]
            μ = μ + φₜ*dy
            𝓦 = 𝓦 + φₜ*φₜ'*dt
        end
    end
    μ, 𝓦
end
