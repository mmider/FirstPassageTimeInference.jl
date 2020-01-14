include(joinpath("..", "FirstPassageTimeInference_for_tests.jl"))

#------------------------------------------------------------------------------#
#                              DATA SIMULATION                                 #
#------------------------------------------------------------------------------#
using Random
θ = [0.3, -5.0, 10.0]
P = OrnsteinUhlenbeck(θ...)
current_prime(t, ::OrnsteinUhlenbeck) = 0.0
parameters = (l=-10.0, L=10.0, dt=0.00001, P=P, N=30, ϵ=0.0)

Random.seed!(5)
current(t, ::OrnsteinUhlenbeck) = 2.0
τ₁ = run_experiment(parameters...)

current(t, ::OrnsteinUhlenbeck) = 4.0
τ₂ = run_experiment(parameters...)

current(t, ::OrnsteinUhlenbeck) = 6.0
τ₃ = run_experiment(parameters...)

current(t, ::OrnsteinUhlenbeck) = 8.0
τ₄ = run_experiment(parameters...)

τᵢs = [τ₁, τ₂, τ₃, τ₄]
trial_lengths = map(x->x[end][2], τᵢs)

τs = begin
    data = [τᵢs[1]]
    for (i, y) in enumerate(trial_lengths[1:end-1])
        append!(data, [map(x->(x[1] + y, x[2] + y), τᵢs[1+i])])
    end
    flatten(data)
end

Xτs = [(parameters.l, parameters.L) for _ in τs]

#------------------------------------------------------------------------------#
#                                INFERENCE                                     #
#------------------------------------------------------------------------------#
using LinearAlgebra

sim_parameters = (
    P = OrnsteinUhlenbeck(0.3, -5.0, 10.0),
    dt = 0.01,
    num_mcmc_steps = 5000,
    ρ = 0.0,
    updt_param_idx = [1, 3],
    t_kernel = RandomWalk([0.6, 1.5, 0.2], [false, false, true]),
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

current(t, ::OrnsteinUhlenbeck) = (
    2 +
    2 * (t >= trial_lengths[1]) +
    2 * (t >= trial_lengths[2]) +
    2 * (t >= trial_lengths[3])
)


(θs, paths), elapsed = @timeit mcmc(τs, Xτs, sim_parameters...)

#------------------------------------------------------------------------------#
#                                  SUMMARY                                     #
#------------------------------------------------------------------------------#

ax = standard_summary_plot(P, paths, θs, [0.3, -5.0, 10.0])
