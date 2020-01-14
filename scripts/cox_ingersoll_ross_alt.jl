include(joinpath("..", "FirstPassageTimeInference_for_tests.jl"))

#------------------------------------------------------------------------------#
#                              DATA SIMULATION                                 #
#------------------------------------------------------------------------------#
θ = [0.1, 10.0, 0.0, 1.0]
P = CoxIngersollRossAlt(θ...)
parameters = (l=1.0, L=10.0, dt=0.00001, P=P, N=100, ϵ=1.0)
current(t) = 0.0#4*sin(π/20.0*t)#0.0#10*sin(π/5.0*t)
current_prime(t) = 0.0 #4*π/20.0*cos(π/20.0*t)#0.0#10*π/5.0*cos(π/5.0*t)

Random.seed!(10)
τs = run_experiment(parameters...)
Xτs = [(parameters.l, parameters.L) for _ in τs]

#------------------------------------------------------------------------------#
#                                INFERENCE                                     #
#------------------------------------------------------------------------------#

parameters = (
    P = CoxIngersollRossAlt(θ...),
    dt =  0.01,
    num_mcmc_steps = 100000,
    ρ = 0.0,
    updt_param_idx = [1, 2, 4],
    t_kernel = CIRRandomWalkAlt([0.3, 1.0, 2.0, 0.25]),
    priors = (
        #MvNormal([0.0,0.0], diagm(0=>[1000.0, 1000.0])),
        ImproperPrior(),
        ImproperPrior(),
        ImproperPrior(),
        ImproperPosPrior(),
    ),
    update_type = (
        #ConjugateUpdate(),
        MetropolisHastings(),
        MetropolisHastings(),
        MetropolisHastings(),
        MetropolisHastings(),
    ),
    save_iter = 1000,
    verb_iter = 100,
)

θs, paths = mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                  SUMMARY                                     #
#------------------------------------------------------------------------------#

ax = standard_summary_plot(P, paths, θs, θ)
