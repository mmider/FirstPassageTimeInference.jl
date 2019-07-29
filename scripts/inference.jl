SRC_DIR = joinpath(Base.source_dir(), "..", "src")
OUT_DIR=joinpath(Base.source_dir(), "..", "output")
mkpath(OUT_DIR)

include(joinpath(SRC_DIR, "reposit.jl"))
include(joinpath(SRC_DIR, "brownian_bridges.jl"))
include(joinpath(SRC_DIR, "Bessel.jl"))
include(joinpath(SRC_DIR, "mcmc.jl"))
include(joinpath(SRC_DIR, "ornstein_uhlenbeck.jl"))

obsTimes = [[0.0, 1.0], [2.0, 3.0], [3.0, 3.5], [4.0, 4.2]]
obsVals = [[1.0, 2.0], [2.5, 3.0], [3.0, 0.5], [3.0, 1.0]]

dt = 0.01
θ = [1.0, 1.0, 0.5]
P = OrnsteinUhlenbeck(θ...)

XX  = mcmc(obsTimes, obsVals, P, dt, 1000, 0.9)

XX.XX

using Plots
p = plot(XX.XX[1].tt, XX.XX[1].yy, label="")
