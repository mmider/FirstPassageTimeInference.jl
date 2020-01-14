struct CoxIngersollRossAlt <: ContinuousTimeProcess{Float64}
    θ::Float64
    α::Float64
    low::Float64
    σ::Float64
end

current(t, ::CoxIngersollRossAlt) = 0.0
current_prime(t, ::CoxIngersollRossAlt) = 0.0
state_space(P::CoxIngersollRossAlt) = MustBeAbove(P.low)

b(t, y::Float64, P::CoxIngersollRossAlt) = P.θ*(P.α-y+P.low) + current(t, P)
σ(t, y::Float64, P::CoxIngersollRossAlt) = P.σ * sqrt(y-P.low)

function ϕ(t, y::Float64, P::CoxIngersollRossAlt)
    cur_prime = current_prime(t, P)
    ∂ₜA = cur_prime != 0.0 ? 2.0/P.σ^2*log(y)*cur_prime : 0.0
    return 0.5*(_b_transf(t, y, P)^2 + _b_transf_prime(t, y, P)) + ∂ₜA
end

function _b_transf(t, y::Float64, P::CoxIngersollRossAlt)
    (2.0*(P.α*P.θ+current(t,P))/P.σ^2-0.5)/y - 0.5*P.θ*y
end

function _b_transf_prime(t, y::Float64, P::CoxIngersollRossAlt)
    (0.5-2.0*(P.α*P.θ+current(t,P))/P.σ^2)/y^2 - 0.5*P.θ
end

A(t, y::Float64, P::CoxIngersollRossAlt) = (2.0*(P.α*P.θ+current(t,P))/P.σ^2-0.5)*log(y)-0.25*P.θ*y^2

η(t, y::Float64, P::CoxIngersollRossAlt) = 2.0/P.σ*sqrt(y-P.low)

η⁻¹(y::Float64, P::CoxIngersollRossAlt) = η⁻¹(nothing, y, P)
η⁻¹(t, y::Float64, P::CoxIngersollRossAlt) = 0.25*(P.σ*y)^2 + P.low

params(P::CoxIngersollRossAlt) = [P.θ, P.α, P.low, P.σ]

clone(P::CoxIngersollRossAlt, θ) = CoxIngersollRossAlt(θ...)
