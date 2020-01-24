struct CoxIngersollRoss <: ContinuousTimeProcess{Float64}
    θ::Float64
    α::Float64
    low::Float64
    σ::Float64
end

current(t, ::CoxIngersollRoss) = 0.0
current_prime(t, ::CoxIngersollRoss) = 0.0
# NOTE this is the state space of the Lamperti transformed diffusion !!!
state_space(P::CoxIngersollRoss) = MustBePositive() #MustBeAbove(P.low)

b(t, y::Float64, P::CoxIngersollRoss) = P.α - P.θ*(y-P.low) + current(t, P)
σ(t, y::Float64, P::CoxIngersollRoss) = P.σ * sqrt(y-P.low)

function ϕ(t, y::Float64, P::CoxIngersollRoss)
    cur_prime = current_prime(t, P)
    ∂ₜA = cur_prime != 0.0 ? 2.0/P.σ^2*log(y)*cur_prime : 0.0
    return 0.5*(_b_transf(t, y, P)^2 + _b_transf_prime(t, y, P)) + ∂ₜA
end

function _b_transf(t, y::Float64, P::CoxIngersollRoss)
    (2.0*(P.α+current(t,P))/P.σ^2-0.5)/y - 0.5*P.θ*y
end

function _b_transf_prime(t, y::Float64, P::CoxIngersollRoss)
    (0.5-2.0*(P.α+current(t,P))/P.σ^2)/y^2 - 0.5*P.θ
end

A(t, y::Float64, P::CoxIngersollRoss) = (2.0*(P.α+current(t,P))/P.σ^2-0.5)*log(y)-0.25*P.θ*y^2

η(t, y::Float64, P::CoxIngersollRoss) = 2.0/P.σ*sqrt(y-P.low)

η⁻¹(y::Float64, P::CoxIngersollRoss) = η⁻¹(nothing, y, P)
η⁻¹(t, y::Float64, P::CoxIngersollRoss) = 0.25*(P.σ*y)^2 + P.low

params(P::CoxIngersollRoss) = [P.θ, P.α, P.low, P.σ]

clone(P::CoxIngersollRoss, θ) = CoxIngersollRoss(θ...)

φ(t, y::Float64, P::CoxIngersollRoss) = @SVector[-0.5*y, 2.0/(P.σ^2*y)]
φᶜ(t, y::Float64, P::CoxIngersollRoss) = (2.0*current(t, P)/P.σ^2-0.5)/y
