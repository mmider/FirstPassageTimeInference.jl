SRC_DIR = joinpath(Base.source_dir(), "..", "src")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "reposit.jl"))
include(joinpath(SRC_DIR, "brownian_bridges.jl"))
include(joinpath(SRC_DIR, "Bessel.jl"))
include(joinpath(SRC_DIR, "integrate.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))#
include(joinpath(SRC_DIR, "random_walk.jl"))
include(joinpath(SRC_DIR, "priors.jl"))
include(joinpath(SRC_DIR, "ornstein_uhlenbeck.jl"))

obsTimes = [[0.0, 1.0], [2.0, 3.0], [3.0, 3.5], [4.0, 4.2]]
obsVals = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]

dt = 0.01
θ = [1.0, 1.0, 0.5]
P = OrnsteinUhlenbeck(θ...)

θs, paths  = mcmc(obsTimes, obsVals, P, dt, 1000, 0.0, 10)


using Plots

p = plot([],[], label="")
N, M = length(paths), length(paths[1])
for i in 1:N
    for j in 1:M
        plot!(paths[i][j].tt, [η⁻¹(x, P) for x in paths[i][j].yy], label="", alpha=0.2, color="steelblue")
    end
end

display(p)
