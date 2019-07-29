SRC_DIR = joinpath(Base.source_dir(), "..", "src")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

using Bridge
using Random

include(joinpath(SRC_DIR, "ornstein_uhlenbeck.jl"))

Wnr = Wiener()
θ = [1.0, 1.0, 0.5]

l₀ = -0.5
L = 0.5
P = OrnsteinUhlenbeck(θ...)

function sampleFPT(x0, xT, dt, P)
    y = x0
    t = 0.0
    sqdt = √dt
    while y < xT
        y += b(t, y, P)*dt + σ(t, y, P)*sqdt*randn(Float64)
        t += dt
    end
    t
end

N, dt = 100, 0.001
Random.seed!(4)
samples = vcat(0.0, cumsum([sampleFPT(l₀, L, dt, P) for i in 1:N]))

obsTimes = collect(zip(samples[1:end-1], samples[2:end]))
obsVals = [(l₀, L) for _ in obsTimes]

obsTimes, obsVals
