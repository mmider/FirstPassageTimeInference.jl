include(joinpath("..", "FirstPassageTimeInference_for_tests.jl"))
include(joinpath("..", "src", "auxiliary", "hodgkin_huxley_model_supporting_functions.jl"))

#------------------------------------------------------------------------------#
#                                Fetch the data
#------------------------------------------------------------------------------#

data = read_τ_data(joinpath(OUT_DIR, "first_passage_times_hodgkin_huxley.csv"))

#------------------------------------------------------------------------------#
#                               Prepare the data
#------------------------------------------------------------------------------#

# trim the data for quicker debugging
data = map(x-> x[1:10], data)

# reformatting
cumulative_trial_lengths = Tuple(cumsum(map(x->x[end][2], data)))
τs = begin
    data_new = [data[1]]
    for (i, y) in enumerate(cumulative_trial_lengths[1:end-1])
        append!(data_new, [map(x->(x[1] + y, x[2] + y), data[1+i])])
    end
    flatten(data_new)
end

Xτs = [(-8.5, 13.0) for _ in τs]

#******************************************************************************#
#------------------------------------------------------------------------------#
#                          Run inference on OU process
#------------------------------------------------------------------------------#
#******************************************************************************#

current(t, ::OrnsteinUhlenbeck) = (
    6.0 +
    2.25 * (t <= cumulative_trial_lengths[3]) +
    2.25 * (t <= cumulative_trial_lengths[2]) +
    2.25 * (t <= cumulative_trial_lengths[1])
)
current_prime(t, ::OrnsteinUhlenbeck) = 0.0

P = OrnsteinUhlenbeck(0.1, 1.0, 1.0)

parameters = (
    P = P,
    dt = 0.1,
    num_mcmc_steps = 1000,
    ρ = 0.7,
    updt_param_idx = [1, 3],
    t_kernel = RandomWalk([0.4, 0.2, 0.3], [false, false, true]),
    priors = (
        MvNormal([0.0, 0.0], diagm(0=>[1000.0, 1000.0])),
        ImproperPosPrior(),
        ImproperPosPrior(),
    ),
    update_type = (
        ConjugateUpdate(),
        MetropolisHastings(),
        MetropolisHastings(),
    ),
    save_iter = 100,
    verb_iter = 100,
)

(θs, paths, estim_θ), elapsed = @timeit mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                   SUMMARY
#------------------------------------------------------------------------------#
ax = standard_summary_plot(P, paths, θs, estim_θ)

"""
    run_randomised_experiment(::T, l, L, dt, N, θs) where T <: ContinuousTimeProcess

Sample first passage times for a diffusion with law of type `T` with parameters
drawn at random from a Markov chain of parameters `θs`---effectively, this
function performs sampling of first passage times from diffusion laws, whose
parameter is integrated against its posterior distribution.
"""
function run_randomised_experiment(::T, l, L, dt, N, θs) where T <: ContinuousTimeProcess
    num_samples = length(θs)
    quarter_θs = div(num_samples, 4)
    θs_to_sample = θs[quarter_θs:end]
    samples = zeros(Float64, N)
    for i in 1:N
        P = T(rand(θs_to_sample)...)
        samples[i] = sampleFPT(0.0, l, L, dt, P)
    end
    samples
end


τ_parameters = (
    P = P,
    l = Xτs[1][1],
    L = Xτs[1][2],
    dt = 0.01,
    num_samples = Int64(1e4),
    θs = θs,
)

τs_OU = map([6.0, 8.25, 10.5, 12.75]) do I_t
    current(t, ::OrnsteinUhlenbeck) = I_t
    run_randomised_experiment(τ_parameters...)
end

axs = plot_many_τ_hist(τs_OU)
# to run the line below you must have executed corresponding lines from
# `hodgkin_huxley_model.jl` that estimate the fpt density
plot_many_τ_hist(τs_HH, nbins=600, ax=axs)

# zoom-in
for ax in axs ax.set_ylim([0.0, 0.04]) end
for ax in axs ax.set_xlim([0, 100]) end

#******************************************************************************#
#------------------------------------------------------------------------------#
#                        Run inference on CIR process
#------------------------------------------------------------------------------#
#******************************************************************************#
Random.seed!(5)
current(t, ::CoxIngersollRoss) = (
    6.0 +
    2.25 * (t <= cumulative_trial_lengths[3]) +
    2.25 * (t <= cumulative_trial_lengths[2]) +
    2.25 * (t <= cumulative_trial_lengths[1])
)
current_prime(t, ::CoxIngersollRoss) = 0.0

# θ = [0.3, 1.5, -80.0, 1.2]
θ = [0.4, 1.0, -20.0, 1.0]
P = CoxIngersollRoss(θ...)
parameters = (
    P = P,
    dt = 0.1,
    num_mcmc_steps = 1000,
    ρ = 0.95,
    updt_param_idx = [1, 2, 3, 4],
    t_kernel = CIRRandomWalk([0.2, 0.2, 1.0, 0.1]),
    priors = (
        ImproperPosPrior(),
        ImproperPrior(),
        NormalPrior(Xτs[1][1], 1.0),
        #BackExp(10.0, Xτs[1][1]),
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

(θs, paths, estim_θ), elapsed = @timeit mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                   SUMMARY
#------------------------------------------------------------------------------#
ax = standard_summary_plot(P, paths, θs, estim_θ)

#******************************************************************************#
#------------------------------------------------------------------------------#
#                     Run inference on Langevin-t process
#                   NOTE this actually will not work in practice...
#                   when we talk, I think I can give a heuristic argument
#                   for why not
#------------------------------------------------------------------------------#
#******************************************************************************#

current(t, ::LangevinT) = (
    6.0 +
    2.25 * (t <= cumulative_trial_lengths[3]) +
    2.25 * (t <= cumulative_trial_lengths[2]) +
    2.25 * (t <= cumulative_trial_lengths[1])
)
current_prime(t, ::LangevinT) = 0.0

P = LangevinT(7.0, -200000.0, 80.0)

parameters = (
    P = P,
    dt = 0.1,
    num_mcmc_steps = 1000,
    ρ = 0.7,
    updt_param_idx = [2, 3],
    t_kernel = RandomWalk([1.0, 1000.0, 0.5], [true, false, true]),
    priors = (
        #MvNormal([0.0, 0.0], diagm(0=>[1000.0, 1000.0])),
        ImproperPosPrior(),
        ImproperPrior(),
        ImproperPosPrior(),
    ),
    update_type = (
        #ConjugateUpdate(),
        MetropolisHastings(),
        MetropolisHastings(),
        MetropolisHastings(),
    ),
    save_iter = 100,
    verb_iter = 100,
)

(θs, paths, estim_θ), elapsed = @timeit mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                   SUMMARY
#------------------------------------------------------------------------------#
ax = standard_summary_plot(P, paths, θs, estim_θ)

τ_parameters = (
    P = P,
    l = Xτs[1][1],
    L = Xτs[1][2],
    dt = 0.01,
    num_samples = Int64(1e4),
    θs = θs,
)

τs_OU = map([6.0, 8.25, 10.5, 12.75]) do I_t
    current(t, ::OrnsteinUhlenbeck) = I_t
    run_randomised_experiment(τ_parameters...)
end

axs = plot_many_τ_hist(τs_OU)
# to run the line below you must have executed corresponding lines from
# `hodgkin_huxley_model.jl` that estimate the fpt density
plot_many_τ_hist(τs_HH, nbins=600, ax=axs)

# zoom-in
for ax in axs ax.set_ylim([0.0, 0.04]) end
for ax in axs ax.set_xlim([0, 100]) end



#******************************************************************************#
#------------------------------------------------------------------------------#
#       Run inference on (yet undefined) OU process with killing intensity
#------------------------------------------------------------------------------#
#******************************************************************************#
