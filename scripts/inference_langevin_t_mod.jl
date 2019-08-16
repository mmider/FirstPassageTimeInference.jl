SRC_DIR = joinpath(Base.source_dir(), "..", "src")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

using Bridge
using Random

# SIMULATE THE DATA
include(joinpath(SRC_DIR, "langevin_t_mod.jl"))

Wnr = Wiener()
θ = [3.0, 0.0, 1.0]
f(t) = 0.0
f_prime(t) = 0.0

P = LangevinTMod(θ..., f, f_prime)

function sampleFPT(t0, x0, xT, dt, P)
    y = x0
    t = t0
    sqdt = √dt
    while y < xT
        y += b(t, y, P)*dt + σ(t, y, P)*sqdt*randn(Float64)
        t += dt
    end
    t
end

function run_experiment(l, L, dt, P, N, ϵ=0.0)
    samples = [(0.0,0.0) for _ in 1:N]
    t0 = 0.0
    for i in 1:N
        T = sampleFPT(t0, l, L, dt, P)
        samples[i] = (t0, T)
        t0 = T+ϵ
    end
    samples
end

N, dt, ϵ = 10, 0.00001, 1.0
Random.seed!(4)
l, L = -10.0, -6.0
samples1 = run_experiment(l, L, dt, P, N, ϵ)
sampleVals1 = [(l, L) for _ in samples1]
l, L = -14.0, -7.0
samples2 = run_experiment(l, L, dt, P, N, ϵ)
sampleVals2 = [(l, L) for _ in samples1]

obsTimes = vcat(samples1, samples2)
obsVals = vcat(sampleVals1, sampleVals2)




# INFERENCE

dt = 0.01
θ = [3.0, 0.0, 1.0]
P = LangevinTMod(θ..., f, f_prime)
tKernel = RandomWalk([0.35, 3.0, 0.6], [true, false, true])
priors = (
          ImproperPosPrior(),
          ImproperPrior(),
          ImproperPosPrior(),
          )
updateType = (
              MetropolisHastings(),
              MetropolisHastings(),
              MetropolisHastings()
              )

numMCMCsteps = 10000
ρ = 0.0
updtParamIdx = [1,2,3]
saveIter = 100
verbIter = 100
start = time()
θs, paths  = mcmc(obsTimes, obsVals, P, dt, numMCMCsteps, ρ, updtParamIdx,
                  tKernel, priors, updateType, saveIter, verbIter)
elapsed = time() - start
print("Elapsed time: ", elapsed, ".")
θs
