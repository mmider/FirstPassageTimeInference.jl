struct OrnsteinUhlenbeck <: ContinuousTimeProcess{Float64}
    θ::Float64
    μ::Float64
    σ::Float64
end

drift(t, y::Float64, P::OrnsteinUhlenbeck) = P.θ*(P.μ-y)
vola(t, y::Float64, P::OrnsteinUhlenbeck) = P.σ

η(y::Float64, P::OrnsteinUhlenbeck) = η(nothing, y, P)
η(t, y::Float64, P::OrnsteinUhlenbeck) = y/P.σ

η⁻¹(y::Float64, P::OrnsteinUhlenbeck) = η⁻¹(nothing, y, P)
η⁻¹(t, y::Float64, P::OrnsteinUhlenbeck) = P.σ*y

params(P::OrnsteinUhlenbeck) = [P.θ, P.μ, P.σ]
