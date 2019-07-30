SRC_DIR = joinpath(Base.source_dir(), "..", "src")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

using Bridge
using Random

include(joinpath(SRC_DIR, "ornstein_uhlenbeck.jl"))

Wnr = Wiener()
θ = [1.0, 1.0, 1.0]#[0.1, 15.0, 1.0]#[1.0, 1.0, 1.0]#

l₀ = -0.5#-0.5#
L = 0.5#2.5#
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

N, dt = 20, 0.0001
Random.seed!(4)
samples = vcat(0.0, cumsum([sampleFPT(l₀, L, dt, P) for i in 1:N]))

obsTimes = collect(zip(samples[1:end-1], samples[2:end]))
obsVals = [(l₀, L) for _ in obsTimes]

l2 = 0.0
L2 = 2.0
samples2 = vcat(0.0, cumsum([sampleFPT(l2, L2, dt, P) for i in 1:N]))
obsTimes2 = collect(zip(samples2[1:end-1], samples2[2:end]))
obsVals2 = [(l2, L2) for _ in obsTimes2]

l3 = 1.5
L3 = 2.5
samples3 = vcat(0.0, cumsum([sampleFPT(l3, L3, dt, P) for i in 1:N]))
obsTimes3 = collect(zip(samples3[1:end-1], samples3[2:end]))
obsVals3 = [(l3, L3) for _ in obsTimes3]


#obsTimes, obsVals = vcat(obsTimes, obsTimes2), vcat(obsVals, obsVals2)
obsTimes, obsVals = vcat(obsTimes, obsTimes2), vcat(obsVals, obsVals2)
obsTimes, obsVals
