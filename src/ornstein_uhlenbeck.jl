struct OrnsteinUhlenbeck <: ContinuousTimeProcess{Float64}
    θ::Float64
    μ::Float64
    σ::Float64
end

b(t, y::Float64, P::OrnsteinUhlenbeck) = P.θ*(P.μ-y)
σ(t, y::Float64, P::OrnsteinUhlenbeck) = P.σ

ϕ(y::Float64, P::OrnsteinUhlenbeck) = ϕ(nothing, y, P)
ϕ(t, y::Float64, P::OrnsteinUhlenbeck) = 0.5*((P.θ*(P.μ/P.σ-y))^2-P.θ)

A(y::Float64, P::OrnsteinUhlenbeck) = A(nothing, y, P)
A(t, y::Float64, P::OrnsteinUhlenbeck) = P.θ*P.μ/P.σ*y - 0.5*P.θ*y^2

η(y::Float64, P::OrnsteinUhlenbeck) = η(nothing, y, P)
η(t, y::Float64, P::OrnsteinUhlenbeck) = y/P.σ

η⁻¹(y::Float64, P::OrnsteinUhlenbeck) = η⁻¹(nothing, y, P)
η⁻¹(t, y::Float64, P::OrnsteinUhlenbeck) = P.σ*y

params(P::OrnsteinUhlenbeck) = [P.θ, P.μ, P.σ]


clone(P::OrnsteinUhlenbeck, θ) = OrnsteinUhlenbeck(θ...)

logD(y::Float64, P::OrnsteinUhlenbeck) = -log(P.σ)
