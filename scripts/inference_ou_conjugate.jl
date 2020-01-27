include(joinpath("..", "FirstPassageTimeInference_for_tests.jl"))

#------------------------------------------------------------------------------#
#                              DATA SIMULATION                                 #
#------------------------------------------------------------------------------#
using Random
θ = [0.1, 1.5, 1.0]
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
using LinearAlgebra

parameters = (
    P = OrnsteinUhlenbeck(0.1, 1.5, 1.0),
    dt = 0.01,
    num_mcmc_steps = 10000,
    ρ = 0.0,
    updt_param_idx = [1, 3],
    t_kernel = RandomWalk([0.4, 0.2, 0.3], [false, false, true]),
    priors = (
        MvNormal([0.0,0.0], diagm(0=>[1000.0, 1000.0])),
        ImproperPrior(),
        ImproperPrior(),
        ),
    update_type = (
        ConjugateUpdate(),
        MetropolisHastings(),
        MetropolisHastings(),
    ),
    save_iter = 10,
    verb_iter = 100,
)

(θs, paths, mean_estim_θ), elapsed  = @timeit mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                  SUMMARY                                     #
#------------------------------------------------------------------------------#

ax = standard_summary_plot(P, paths, θs, [0.1, 1.5, 1.0])
