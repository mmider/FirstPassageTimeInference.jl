include(joinpath("..", "FirstPassageTimeInference_for_tests.jl"))

#------------------------------------------------------------------------------#
#                              DATA SIMULATION                                 #
#------------------------------------------------------------------------------#
using Random
θ = [0.1, 1.5, 1.0]
P = OrnsteinUhlenbeck(θ...)
current(t) = 0.5*sin(π/20.0*t)#0.0#10*sin(π/5.0*t)
current_prime(t) = 0.5*π/20.0*cos(π/20.0*t)#0.0#10*π/5.0*cos(π/5.0*t)

parameters = [
    (l=0.0, L=10.0, dt=0.00001, P=P, N=40, ϵ=0.0),
    (l=0.0, L=5.0, dt=0.00001, P=P, N=40, ϵ=0.0),
    (l=0.0, L=8.0, dt=0.00001, P=P, N=40, ϵ=0.0),
]

Random.seed!(5)
τs = flatten([run_experiment(p...) for p in parameters])
Xτs = flatten([[(p.l, p.L) for _ in 1:p.N] for p in parameters])

#------------------------------------------------------------------------------#
#                                INFERENCE                                     #
#------------------------------------------------------------------------------#
using LinearAlgebra

parameters = (
    P = OrnsteinUhlenbeck(0.4, 3.0, 3.0),
    dt = 0.01,
    num_mcmc_steps = 10000,
    ρ = 0.0,
    updt_param_idx = [1, 3],
    t_kernel = RandomWalk([0.6, 1.5, 0.4], [false, false, true]),
    priors = (
        MvNormal([0.0,0.0], diagm(0=>[1000.0, 1000.0])),
        ImproperPrior(),
        ImproperPosPrior(),
    ),
    update_type = (
        ConjugateUpdate(),
        MetropolisHastings(),
        MetropolisHastings()
    ),
    save_iter = 100,
    varb_iter = 100,
)

(θs, paths), elapsed = @timeit mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                  SUMMARY                                     #
#------------------------------------------------------------------------------#

ax = standard_summary_plot(P, paths, θs, [0.1, 1.5, 1.0])
ax[2].set_ylim([-0.05, 0.2])
ax[3].set_ylim([0.8, 2.0])
ax[4].set_ylim([0.7, 1.2])


#=
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
=#
