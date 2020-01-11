include(joinpath("..", "FirstPassageTimeInference_for_tests.jl"))

#------------------------------------------------------------------------------#
#                              DATA SIMULATION                                 #
#------------------------------------------------------------------------------#
using Random
θ = [0.1, 15.0, 1.0]
P = OrnsteinUhlenbeck(θ...)
parameters = [
    (l=0.0, L=10.0, dt=0.00001, P=P, N=30, ϵ=0.0),
    (l=0.0, L=2.0, dt=0.00001, P=P, N=30, ϵ=0.0),
    (l=1.0, L=2.5, dt=0.00001, P=P, N=30, ϵ=0.0),
    ]
Random.seed!(4)

τs = flatten([run_experiment(p...) for p in parameters])
Xτs = flatten([[(p.l, p.L) for _ in 1:p.N] for p in parameters])


#------------------------------------------------------------------------------#
#                                INFERENCE                                     #
#------------------------------------------------------------------------------#

parameters = (
    P = OrnsteinUhlenbeck(0.1, 15.0, 1.0),
    dt = 0.01,
    num_mcmc_steps = 30000,
    ρ = 0.0,
    updt_param_idx = [1, 2, 3],
    t_kernel = RandomWalk([0.35, 3.0, 0.6], [true, false, true]),
    priors = (
        Normal(0.0, 30.0),
        Normal(0.0, 30.0),
        ImproperPosPrior(),
    ),
    update_type = (
        MetropolisHastings(),
        MetropolisHastings(),
        MetropolisHastings()
    ),
    save_iter = 100,
    verb_iter = 100,
)


(θs, paths), elapsed  = @timeit mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                  SUMMARY                                     #
#------------------------------------------------------------------------------#

ax = standard_summary_plot(P, paths, θs, θ)