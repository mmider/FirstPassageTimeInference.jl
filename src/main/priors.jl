"""
    ImproperPrior
Flat prior
"""
struct ImproperPrior end
logpdf(::ImproperPrior, θ) = 0.0

struct ImproperPosPrior end
logpdf(::ImproperPosPrior, θ) = -log(θ)
