#==============================================================================#
#
#       Modification of the Cox-Ingersoll Ross model from ℝ₊ to ℝ
#
#       NOTE it may seem artifical at first, but it is essentially
#       a more natural definition of the process for modelling phenomena
#       with no "hard-zero", so-to-say, but where the process must still be
#       repelled from one side disproportionatelly stronger than from another.
#       The advantage of using such a process is that the usual problems
#       regarding finite diffusion boundary---that appear in numerical
#       analysis of Cox-Ingersoll Ross process---simply disappear
#       TODO I wanted to implement this, but it turns out that performing
#       closed-form calculations with this is more difficult than anticipated,
#       if not straight out impossible, this will need to be revised with
#       guided proposals for elliptic diffusions...
#
#==============================================================================#

struct NaturalCIR <: ContinuousTimeProcess{Float64}
    θ::Float64
    μ::Float64
    k::Float64
    s::Float64
    σ::Float64
end

state_space(::NaturalCIR) = Unrestricted()

# default values of the `smooth` current
current(t, ::NaturalCIR) = 0.0
current_prime(t, ::NaturalCIR) = 0.0

b(t, y::Float64, P::NaturalCIR) = P.θ * (P.μ - y) + current(t, P)
σ(t, y::Float64, P::NaturalCIR) = P.σ / (1.0 + exp( -P.s*(y-P.k) ))

tt = 0.0:0.01:10.0
P = NaturalCIR(1.0, 0.2, 0.0, 10.0, 1.0)
drift = b
XX = _simulate(Float64, 0.0, tt, P)
using PyPlot
plt.plot(tt, XX)

function ϕ(t, y::Float64, P::NaturalCIR)
    cur_prime = current_prime(t, P)
    ∂ₜA = cur_prime != 0.0 ? ... : 0.0
    0.5 * (_b_transf(t, y, P)^2 + _b_transf_prime(t, y, P))
end

function _b_transf(t, y::Float64, P::CoxIngersollRossAlt)
    (2.0*(P.α*P.θ+current(t,P))/P.σ^2-0.5)/y - 0.5*P.θ*y
end

function _b_transf_prime(t, y::Float64, P::CoxIngersollRossAlt)
    (0.5-2.0*(P.α*P.θ+current(t,P))/P.σ^2)/y^2 - 0.5*P.θ
end

A(t, y::Float64, P::CoxIngersollRossAlt) = (2.0*(P.α*P.θ+current(t,P))/P.σ^2-0.5)*log(y)-0.25*P.θ*y^2


η(t, y::Float64, P::NaturalCIR) = (y - exp( -(y-P.k) ) + exp(k)) / P.σ

# this is not that important, don't perform any inversions
η⁻¹(t, y::Float64, P::NaturalCIR) = y
