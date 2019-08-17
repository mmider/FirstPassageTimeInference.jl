SRC_DIR = joinpath(Base.source_dir(), "..", "src")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

using Bridge
using Random

# SIMULATE THE DATA
include(joinpath(SRC_DIR, "ornstein_uhlenbeck_conjug_mod.jl"))

Wnr = Wiener()
θ = [0.1, 1.5, 1.0]
f(t) = 0.5*sin(π/20.0*t)#0.0#10*sin(π/5.0*t)
f_prime(t) = 0.5*π/20.0*cos(π/20.0*t)#0.0#10*π/5.0*cos(π/5.0*t)

P = OrnsteinUhlenbeckMod(θ..., f, f_prime)

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

N, dt, ϵ = 40, 0.000001, 1.0
Random.seed!(5)
l, L = 0.0, 10.0
samples1 = run_experiment(l, L, dt, P, N, ϵ)
sampleVals1 = [(l, L) for _ in samples1]
l, L = 0.0, 5.0
samples2 = run_experiment(l, L, dt, P, N, ϵ)
sampleVals2 = [(l, L) for _ in samples1]
l, L = 5.0, 8.0
samples3 = run_experiment(l, L, dt, P, N, ϵ)
sampleVals3 = [(l, L) for _ in samples1]

obsTimes = vcat(samples1, samples2, samples3)
obsVals = vcat(sampleVals1, sampleVals2, sampleVals3)


include(joinpath(SRC_DIR, "reposit.jl"))
include(joinpath(SRC_DIR, "brownian_bridges.jl"))
include(joinpath(SRC_DIR, "bessel.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))
include(joinpath(SRC_DIR, "random_walk.jl"))
include(joinpath(SRC_DIR, "priors.jl"))

# INFERENCE

dt = 0.01
θ = [0.4, 3.0, 3.0]
P = OrnsteinUhlenbeckMod(θ..., f, f_prime)
tKernel = RandomWalk([0.6, 1.5, 0.4], [false, false, true])
#tKernel = RandomWalk([0.35, 3.0, 0.6], [true, false, true])
priors = (
          MvNormal([0.0,0.0], diagm(0=>[1000.0, 1000.0])),
          ImproperPrior(),
          ImproperPosPrior(),
          )
updateType = (
              ConjugateUpdate(),
              MetropolisHastings(),
              MetropolisHastings()
              )

numMCMCsteps = 10000
ρ = 0.0
updtParamIdx = [1,3]
saveIter = 100
verbIter = 100
start = time()
θs, paths  = mcmc(obsTimes, obsVals, P, dt, numMCMCsteps, ρ, updtParamIdx,
                  tKernel, priors, updateType, saveIter, verbIter)
elapsed = time() - start
print("Elapsed time: ", elapsed, ".")
θs

using Plots
p = plot([],[], label="")
N, M = length(paths), length(paths[1])
for i in max(1,N-100):N
    for j in 1:min(M,6)
        plot!(paths[i][j].tt, [x for x in paths[i][j].yy], label="",
              alpha=0.2, color="steelblue")
    end
end

display(p)


plot([θ[1] for θ in θs])
plot([θ[2] for θ in θs])
plot([θ[3] for θ in θs])

using DataFrames
using CSV

#cd("../firstPassageTimeInference.jl/output")
df = DataFrame([[θ[i] for θ in θs] for i in 1:3])
CSV.write("inference_OU_mod_theta.csv", df)

for j in 1:5
    df = DataFrame(hcat([hcat(paths[i][j].tt, paths[i][j].yy) for i in 1:100]...))
    CSV.write("inference_OU_mod_path_" * string(j) * ".csv", df)
end

df = DataFrame(obs=[v[2]-v[1] for v in obsTimes])
CSV.write("inference_OU_mod_obs.csv", df)
