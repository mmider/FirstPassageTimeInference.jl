SRC_DIR = joinpath(Base.source_dir(), "..", "src")
OUT_DIR = joinpath(Base.source_dir(), "..", "output")
SCRIPT_DIR = joinpath(Base.source_dir(), "..", "scripts")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "reposit.jl"))
include(joinpath(SRC_DIR, "brownian_bridges.jl"))
include(joinpath(SRC_DIR, "Bessel.jl"))
include(joinpath(SRC_DIR, "integrate.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))
include(joinpath(SRC_DIR, "random_walk.jl"))
include(joinpath(SRC_DIR, "priors.jl"))
include(joinpath(SRC_DIR, "ornstein_uhlenbeck.jl"))


obsTimes, obsVals = include(joinpath(SCRIPT_DIR, "simulate_data.jl"))

dt = 0.01
θ = [1.0, 1.0, 1.0]#[0.1, 15.0, 1.0]#[0.1, 5.0, 0.5]
P = OrnsteinUhlenbeck(θ...)

tKernel = RandomWalk([0.4, 0.3, 0.2], [false, false, true])
priors = (
          ImproperPrior(),
          ImproperPrior(),
          ImproperPosPrior(),
          )
updateType = (
              MetropolisHastings(),
              MetropolisHastings(),
              MetropolisHastings())
θs, paths  = mcmc(obsTimes, obsVals, P, dt, 30000, 0.7, [1, 2, 3], tKernel,
                  priors, updateType, 10, 100)

using Plots

p = plot([],[], label="")
N, M = length(paths), length(paths[1])
for i in max(1,N-100):N
    for j in 1:min(M,6)
        plot!(paths[i][j].tt, [η⁻¹(x, P) for x in paths[i][j].yy], label="",
              alpha=0.2, color="steelblue")
    end
end

display(p)


plot([θ[1] for θ in θs])
plot([θ[2] for θ in θs])
plot([θ[3] for θ in θs])
