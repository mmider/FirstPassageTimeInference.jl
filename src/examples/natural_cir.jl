struct NaturalCIR <: ContinuousTimeProcess{Float64}
    θ::Float64
    μ::Float64
    k::Float64
    σ::Float64
end

current(t, ::NaturalCIR) = 0.0
current_prime(t, ::NaturalCIR) = 0.0
state_space(::NaturalCIR) = Unrestricted()

b(t, y::Float64, P::NaturalCIR) = P.θ * (P.μ - y) + current(t, P)
σ(t, y::Float64, P::NaturalCIR) = P.σ / (1.0 + exp( -(y-P.k) ))

function ϕ(t, y::Float64, P::NaturalCIR)

end


η(t, y::Float64, P::NaturalCIR) = (y - exp( -(y-P.k) ) + exp(k)) / P.σ

# this is not that important, don't perform any inversions
η⁻¹(t, y::Float64, P::NaturalCIR) = y
