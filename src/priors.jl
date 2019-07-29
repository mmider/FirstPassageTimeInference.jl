"""
    ImproperPrior
Flat prior
"""
struct ImproperPrior end
logpdf(::ImproperPrior, θ) = 0.0
