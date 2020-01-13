include(joinpath("..", "FirstPassageTimeInference_for_tests.jl"))

#------------------------------------------------------------------------------#
#                                Fetch the data
#------------------------------------------------------------------------------#

function read_τ_data(filename)
    datasets = nothing
    num_datasets = nothing
    open(filename, "r") do f
        for (i, line) in enumerate(eachline(f))
            if i == 1
                num_datasets = div(length(split(line, ",")), 2)
                datasets = [Vector{Tuple{Float64, Float64}}() for _ in 1:num_datasets]
            else
                data_line = map(x->parse(Float64, x), split(line, ","))
                for j in 1:num_datasets
                    append!(datasets[j], [(data_line[2*j-1], data_line[2*j])])
                end
            end
        end
    end
    datasets
end

data = read_τ_data(joinpath(OUT_DIR, "first_passage_times_hodgkin_huxley.csv"))

#------------------------------------------------------------------------------#
#                               Prepare the data
#------------------------------------------------------------------------------#

#data = map(x-> x[1:10], data)
trial_lengths = map(x->x[end][2], data[1:end])

τs = begin
    data_new = [data[1]]
    for (i, y) in enumerate(trial_lengths[1:end-1])
        append!(data_new, [map(x->(x[1] + y, x[2] + y), data[1+i])])
    end
    flatten(data_new)
end

Xτs = [(-10.0, 10.0) for _ in τs]

#******************************************************************************#
#------------------------------------------------------------------------------#
#                          Run inference on OU process
#------------------------------------------------------------------------------#
#******************************************************************************#

current(t, ::OrnsteinUhlenbeckMod) = (
    2 +
    2 * (t <= trial_lengths[3]) +
    2 * (t <= trial_lengths[2]) +
    2 * (t <= trial_lengths[1])
)
current_prime(t, ::OrnsteinUhlenbeckMod) = 0.0

P = OrnsteinUhlenbeckMod(0.1, 1.0, 1.0)

parameters = (
    P = P,
    dt = 0.1,
    num_mcmc_steps = 10000,
    ρ = 0.0,
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

(θs, paths), elapsed = @timeit mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                   SUMMARY
#------------------------------------------------------------------------------#
ax = standard_summary_plot(P, paths, θs, [0.3, -10.0, 10.0])

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
    l = -10.0,
    L = 10.0,
    dt = 0.01,
    num_samples = 10000,
    θs = θs,
)

current(t, ::OrnsteinUhlenbeckMod) = 8.0
τ₁_OU = run_randomised_experiment(τ_parameters...)

current(t, ::OrnsteinUhlenbeckMod) = 6.0
τ₂_OU = run_randomised_experiment(τ_parameters...)

current(t, ::OrnsteinUhlenbeckMod) = 4.0
τ₃_OU = run_randomised_experiment(τ_parameters...)

current(t, ::OrnsteinUhlenbeckMod) = 2.0
τ₄_OU = run_randomised_experiment(τ_parameters...)


# to execute lines below, run lines 220--254 in `hodgkin_huxley_model.jl`
fig, ax = plt.subplots(2, 2, figsize=(15, 15))
plot_τ_hist(τ₁, ax[1,1])
plot_τ_hist(τ₂, ax[1,2])
plot_τ_hist(τ₃, ax[2,1])
plot_τ_hist(τ₄, ax[2,2])

plot_τ_hist(τ₁_OU, ax[1,1])
plot_τ_hist(τ₂_OU, ax[1,2])
plot_τ_hist(τ₃_OU, ax[2,1])
plot_τ_hist(τ₄_OU, ax[2,2])


#******************************************************************************#
#------------------------------------------------------------------------------#
#                        Run inference on CIR process
#------------------------------------------------------------------------------#
#******************************************************************************#

current(t, ::CoxIngersollRoss) = (
    2 +
    2 * (t <= trial_lengths[3]) +
    2 * (t <= trial_lengths[2]) +
    2 * (t <= trial_lengths[1])
)
current_prime(t, ::CoxIngersollRoss) = 0.0

θ = [0.1, 3.0, -20.0, 1.0]
P = CoxIngersollRoss(θ...)

parameters = (
    P = P,
    dt = 0.1,
    num_mcmc_steps = 10000,
    ρ = 0.0,
    updt_param_idx = [1, 2, 3, 4],
    t_kernel = RandomWalk([0.4, 0.2, 0.3, 0.3], [true, false, false, true]),
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

(θs, paths), elapsed = @timeit mcmc(τs, Xτs, parameters...)

#------------------------------------------------------------------------------#
#                                   SUMMARY
#------------------------------------------------------------------------------#
ax = standard_summary_plot(P, paths, θs, [0.3, -10.0, 10.0])



#******************************************************************************#
#------------------------------------------------------------------------------#
#                     Run inference on Langevin-t process
#------------------------------------------------------------------------------#
#******************************************************************************#


#******************************************************************************#
#------------------------------------------------------------------------------#
#       Run inference on (yet undefined) OU process with killing intensity
#------------------------------------------------------------------------------#
#******************************************************************************#
