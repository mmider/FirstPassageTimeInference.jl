include(joinpath("..", "FirstPassageTimeInference_for_tests.jl"))


#------------------------------------------------------------------------------#
#                              DATA SIMULATION                                 #
#------------------------------------------------------------------------------#
using Random
θ = [5.0, 0.0, 1.0]
f(t) = 0.0#10*sin(π/5.0*t)
f_prime(t) = 0.0#10*π/5.0*cos(π/5.0*t)
P = LangevinTMod(θ..., f, f_prime)
parameters = [
    (l=-10.0, L=-6.0, dt=0.000001, P=P, N=40, ϵ=0.0),
    (l=-14.0, L=-6.0, dt=0.000001, P=P, N=40, ϵ=0.0),
    (l=-24.0, L=-20.0, dt=0.000001, P=P, N=40, ϵ=0.0),
]
Random.seed!(4)

τs = run_experiment(parameters[1]...)
Xτs = [(parameters[1].l, parameters[1].L) for _ in 1:length(τs)]

#------------------------------------------------------------------------------#
#                                INFERENCE                                     #
#------------------------------------------------------------------------------#
using LinearAlgebra

parameters = (
    P = LangevinTMod(20.0, 0.0, 4.0, f, f_prime),
    dt = 0.01,
    num_mcmc_iter = 10000,
    ρ = 0.0,
    updt_param_idx = [1, 3],
    t_kernel = RandomWalk([1.0, 10.0, 0.7], [true, false, true]),
    priors = (
        ImproperPosPrior(),
        ImproperPrior(),
        ImproperPosPrior(),
    ),
    update_type = (
        MetropolisHastings(),
        MetropolisHastings(),
        MetropolisHastings(),
    ),
    save_iter = 100,
    verb_iter = 100,
)

(θs, paths), elapsed  = @timeit mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                  SUMMARY                                     #
#------------------------------------------------------------------------------#

ax = standard_summary_plot(P, paths, θs, [5.0, 0.0, 1.0])

#=
using DataFrames
using CSV

#cd("../firstPassageTimeInference.jl/output")
df = DataFrame([[θ[i] for θ in θs] for i in 1:3])
CSV.write("inference_langevin_t_n1_n3.csv", df)

for j in 1:5
    df = DataFrame(hcat([hcat(paths[i][j].tt, paths[i][j].yy) for i in 1:100]...))
    CSV.write("inference_langevin_t_path_" * string(j) * "_n1_n3.csv", df)
end

df = DataFrame(obs=[v[2]-v[1] for v in obsTimes])
CSV.write("inference_langevin_t_obs_n1_n3.csv", df)
=#
