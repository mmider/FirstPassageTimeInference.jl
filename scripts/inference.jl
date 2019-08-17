SRC_DIR = joinpath(Base.source_dir(), "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
SCRIPT_DIR = joinpath(Base.source_dir(), "..", "scripts")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "reposit.jl"))
include(joinpath(SRC_DIR, "brownian_bridges.jl"))
include(joinpath(SRC_DIR, "bessel.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))
include(joinpath(SRC_DIR, "random_walk.jl"))
include(joinpath(SRC_DIR, "priors.jl"))
include(joinpath(SRC_DIR, "ornstein_uhlenbeck.jl"))


obsTimes, obsVals = include(joinpath(SCRIPT_DIR, "simulate_data.jl"))

dt = 0.01
θ = [0.1, 15.0, 1.0]#[0.3, 15.0, 3.0]#[0.5, 0.5, 0.5]#[1.0, 1.0, 1.0]#[0.1, 15.0, 1.0]#[0.1, 5.0, 0.5]
P = OrnsteinUhlenbeck(θ...)

tKernel = RandomWalk([0.35, 3.0, 0.6], [true, false, true])
priors = (
          #ImproperPosPrior(),
          #ImproperPrior(),
          Normal(0.0, 30.0),
          Normal(0.0, 30.0),
          #MvNormal([0.0], diagm(0=>[1000.0])),
          ImproperPosPrior(),
          )
updateType = (
              MetropolisHastings(),
              MetropolisHastings(),
              MetropolisHastings()
              )
numMCMCsteps = 30000
ρ = 0.0
updtParamIdx = [1,2,3]
saveIter = 100
verbIter = 100
start = time()
θs, paths  = mcmc(obsTimes, obsVals, P, dt, numMCMCsteps, ρ, updtParamIdx,
                  tKernel, priors, updateType, saveIter, verbIter)
elapsed = time() - start
print("Elapsed time: ", elapsed, ".")
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

"""
using DataFrames
using CSV

cd("../firstPassageTimeInference.jl/output")
df = DataFrame([[θ[i] for θ in θs] for i in 1:3])
CSV.write("inference_OU_theta_n2_n3.csv", df)

for j in 1:5
    df = DataFrame(hcat([hcat(paths[i][j].tt, paths[i][j].yy) for i in 1:100]...))
    CSV.write("inference_OU_path_" * string(j) * "_n2_n3.csv", df)
end

df = DataFrame(obs=[v[2]-v[1] for v in obsTimes])
CSV.write("inference_OU_obs_n2_n3.csv", df)
"""
