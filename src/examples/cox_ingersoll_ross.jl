struct CoxIngersollRoss <: ContinuousTimeProcess{Float64}
    θ::Float64
    α::Float64
    low::Float64
    σ::Float64
end

state_space(P::CoxIngersollRoss) = MustBeAbove(P.low)

b(t, y::Float64, P::CoxIngersollRoss) = P.α - P.θ*(y-P.low)
σ(t, y::Float64, P::CoxIngersollRoss) = P.σ * sqrt(y-P.low)

function ϕ(t, y::Float64, P::CoxIngersollRoss)
    0.5*(_b_transf(t, y, P)^2 + _b_transf_prime(t, y, P))
end

function _b_transf(t, y::Float64, P::CoxIngersollRoss)
    (2.0*P.α/P.σ^2-0.5)/y - 0.5*P.θ*y
end

function _b_transf_prime(t, y::Float64, P::CoxIngersollRoss)
    (0.5-2.0*P.α/P.σ^2)/y^2 - 0.5*θ
end

A(t, y::Float64, P::CoxIngersollRoss) = (2.0*P.α/P.σ^2-0.5)*log(y)-0.25*P.θ*y^2

η(t, y::Float64, P::CoxIngersollRoss) = 2.0/P.σ*sqrt(y-P.low)

η⁻¹(y::Float64, P::CoxIngersollRoss) = η⁻¹(nothing, y, P)
η⁻¹(t, y::Float64, P::CoxIngersollRoss) = 0.25*(P.σ*y)^2 + P.low

params(P::CoxIngersollRoss) = [P.θ, P.α, P.low, P.σ]

clone(P::CoxIngersollRoss, θ) = CoxIngersollRoss(θ...)
