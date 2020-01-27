include(joinpath("..", "FirstPassageTimeInference_for_tests.jl"))

#------------------------------------------------------------------------------#
#                              DATA SIMULATION                                 #
#------------------------------------------------------------------------------#
θ = [0.1, 3.0, 0.0, 1.0]
P = CoxIngersollRoss(θ...)
parameters = (l=10.0, L=20.0, dt=0.00001, P=P, N=40, ϵ=1.0)

τs = run_experiment(parameters...)
Xτs = [(parameters.l, parameters.L) for _ in τs]

#------------------------------------------------------------------------------#
#                                INFERENCE                                     #
#------------------------------------------------------------------------------#

parameters = (
    P = CoxIngersollRoss(θ...),
    dt =  0.01,
    num_mcmc_steps = 10000,
    ρ = 0.0,
    updt_param_idx = [1, 2, 3, 4],
    t_kernel = CIRRandomWalk([0.2, 0.2, 2.0, 0.5]),
    priors = (
        ImproperPosPrior(),
        ImproperPrior(),
        ImproperPrior(),
        ImproperPosPrior(),
    ),
    update_type = (
        MetropolisHastings(),
        MetropolisHastings(),
        MetropolisHastings(),
        MetropolisHastings(),
    ),
    save_iter = 100,
    verb_iter = 100,
)

θs, paths, mean_estim_θ = mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                  SUMMARY                                     #
#------------------------------------------------------------------------------#

ax = standard_summary_plot(P, paths, θs, θ)
