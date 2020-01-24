"""
    ImproperPrior
Flat prior
"""
struct ImproperPrior end
logpdf(::ImproperPrior, θ) = 0.0

struct ImproperPosPrior end
logpdf(::ImproperPosPrior, θ) = -log(θ)

struct NormalPrior
    μ::Float64
    σ::Float64
end
logpdf(p::NormalPrior, θ) = -0.5*log(2.0*π) - log(p.σ) - 0.5*(θ-p.μ)^2/p.σ^2

struct BackExp
    λ::Float64
    cutoff::Float64
end
logpdf(p::BackExp, θ) = θ > p.cutoff ? -Inf : log(p.λ) + p.λ*(θ-p.cutoff)
