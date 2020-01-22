using StaticArrays
using LinearAlgebra
using Parameters
const ℝ{N} = SArray{Tuple{N},Float64,1,N} where N

#==============================================================================#
#
#           Hodkin-Huxley model with stochastic synaptic input
#
#==============================================================================#

abstract type HodgkinHuxley end

struct OU_params
    θ::Float64
    μ::Float64
    σ::Float64
end

@with_kw struct HodgkinHuxleySSI <: HodgkinHuxley # SSI: stochastic synaptic input
    # ------------
    # membrane capacitance
    c_m::Float64    = 1.0
    # ------------
    # maximum conductances per unit area:
    g_K::Float64    = 36.0      # for potassium
    g_Na::Float64   = 120.0     # sodium
    g_l::Float64    = 0.3       # and leak

    # ------------
    # equilibrium potentials
    v_K::Float64    = -12.0     # for potassium
    v_Na::Float64   = 115.0     # sodium
    v_l::Float64    = 10.0      # leak
    v_E::Float64    # excitatory synaptic input
    v_I::Float64    # inhibitory synaptic input

    # -----------
    # total membrane area
    membrane_area::Float64  = 1.0

    # ----------
    # parameters of the synaptic input
    E::OU_params # excitatory
    I::OU_params # inhibitory
end

resting_HH = (v0 = 0.0, n0 = 0.3177, m0 = 0.0529, h0 = 0.5961)

function synaptic_current(t, y::ℝ{6}, P::HodgkinHuxleySSI)
    ( y[5]*(P.v_E - y[1]) + y[6]*(P.v_I - y[1]) ) / P.membrane_area
end

function standard_HH_currents(t, y, P::T) where {T<:HodgkinHuxley}
    ( P.g_K * y[2]^4*(P.v_K-y[1])
     + P.g_Na * y[3]^3*y[4]*(P.v_Na-y[1])
     + P.g_l * (P.v_l-y[1]) )
end

drift(y::Float64, p::OU_params) = p.θ*(p.μ-y)

function drift(t, y::ℝ{6}, P::HodgkinHuxleySSI)
    @SVector[(synaptic_current(t, y, P) + standard_HH_currents(t, y, P))/P.c_m,
              α_n(y[1]) * (1-y[2]) - β_n(y[1]) * y[2],
              α_m(y[1]) * (1-y[3]) - β_m(y[1]) * y[3],
              α_h(y[1]) * (1-y[4]) - β_h(y[1]) * y[4],
              drift(y[5], P.E),
              drift(y[6], P.I)]
end

σ(t, y::ℝ{6}, P::HodgkinHuxleySSI) = @SMatrix[ 0.0 0.0;
                                               0.0 0.0;
                                               0.0 0.0;
                                               0.0 0.0;
                                               P.E.σ 0.0;
                                               0.0 P.I.σ ]

function α_n(y::Float64)
    x = 0.1*(10.0 - y)
    0.1*x/(exp(x)-1.0)
end

function α_m(y::Float64)
    x = 0.1*(25.0 - y)
    x/(exp(x)-1.0)
end

α_h(y::Float64) = 0.07*exp(-0.05*y)

β_n(y::Float64) = 0.125*exp(-0.0125*y)

β_m(y::Float64) = 4.0*exp(-y/18.0)

β_h(y::Float64) = 1.0/(exp(3.0-0.1*y)+1.0)


#==============================================================================#
#
#         Some parametrisations from A. Destexhe et al. 2001
#
#==============================================================================#

layer_VI = (
    area = 34636.0,
    R_in = 58.9,
    E = OU_params(1.0/2.7, 0.012, 0.003),
    I = OU_params(1.0/10.5, 0.057, 0.0066)
)

layer_III = (
    area = 20321.0,
    R_in = 94.2,
    E = OU_params(1.0/7.8, 0.006, 0.0019),
    I = OU_params(1.0/8.8, 0.044, 0.0069)
)

layer_Va = (
    area = 55017.0,
    R_in = 38.9,
    E = OU_params(1.0/2.6, 0.018, 0.035),
    I = OU_params(1.0/8.0, 0.098, 0.0092)
)

layer_Vb = (
    area = 93265.0,
    R_in = 23.1,
    E = OU_params(1.0/2.8, 0.029, 0.0042),
    I = OU_params(1.0/8.5, 0.16, 0.01)
)

artificial = (
    E = OU_params(1.0/2.0, 0.18, 0.0125),
    I = OU_params(1.0/8.0, 0.1125, 0.00125),
)

artificial_excitatory = (
    E = OU_params(1.0/2.0, 0.11, 0.0125),
    I = OU_params(1.0/8.0, 0.0, 0.0),
)

#==============================================================================#
#
#                           Visualisations
#
#==============================================================================#

function _simulate(noise_type::Type{T}, y0, tt, P) where T
    N = length(tt)
    noise = randn(T, N-1)
    XX = zeros(typeof(y0), N)
    XX[1] = y0
    for i in 1:N-1
        x = XX[i]
        dt = tt[i+1] - tt[i]
        XX[i+1] = x + drift(tt[i], x, P) * dt + sqrt(dt) * σ(tt[i], x, P) * noise[i]
    end
    XX
end


# Pick out a type of the neuron for simulations
neuron = artificial

P_HH = HodgkinHuxleySSI(
    #= modifying the default values of the Hodgkin-Huxley model
    c_m = layer_VI.area/1e6,
    g_l = 0.01559,
    g_K = 3.4636,
    g_Na = 2.4245,
    # end of default values modifications =#
    v_E = 75.0,
    v_I = 0.0,
    E = neuron.E,
    I = neuron.I
)

T = 500.0
tt = 0.0:0.0001:T
y0 = ℝ{6}(resting_HH..., neuron.E.μ, neuron.I.μ)
XX = _simulate(ℝ{2}, y0, tt, P_HH)

using PyPlot

function quick_plot(tt, XX, skip=100, reset_lvl=-9.9, threshold=13.0)
    # Compute the effective synaptic current:
    I_t = map( (t,y) -> synaptic_current(t, y, P_HH), tt, XX)

    fig, ax = plt.subplots(7,1, figsize=(15,10), sharex=true)
    for i in 1:6
        ax[i].plot(tt[1:skip:end], map(x->x[i], XX[1:skip:end]))
    end
    ax[7].plot(tt[1:skip:end], I_t[1:skip:end], color="green")

    for (i, name) in enumerate(["memb. potential", "Potassium", "Sodium", "Leak",
                                "Excitatory", "Inhibitory", "Net current"])
        ax[i].set_ylabel(name)
    end
    ax[1].plot([0, T], [reset_lvl, reset_lvl], linestyle="dashed", color="red")
    ax[1].plot([0, T], [threshold, threshold], linestyle="dashed", color="green")

    plt.tight_layout()
end

quick_plot(tt, XX)







#==============================================================================#
#
#                           THE MAIN EXPERIMENT
#           NOTE this is deprecated due to change of HH model above
#
#==============================================================================#

function _simulate_fpt(t0, x0, xT, dt, P)
    y = x0
    t = t0
    sqdt = √dt
    data_type = typeof(x0)
    while y[1] < xT
        y += b(t, y, P) * dt + sqdt * σ(t, y, P) * randn(data_type)
        t += dt
    end
    y, t
end

function _reset_fpt(t0, x0, xT, dt, P)
    y = x0
    t = t0
    sqdt = √dt
    data_type = typeof(x0)
    while y[1] > xT
        y += b(t, y, P) * dt + sqdt * σ(t, y, P) * randn(data_type)
        t += dt
    end
    y, t
end
function run_experiment(t0, x0, reset_lvl, threshold, dt, P, num_obs)
    y =  x0
    t = t0
    obs_times = zeros(Float64, num_obs)
    down_cross_times = zeros(Float64, num_obs)
    obs_counter = 0
    while obs_counter < num_obs
        y, t = _simulate_fpt(t, y, threshold, dt, P)
        obs_counter += 1
        obs_times[obs_counter] = t
        y, t = _reset_fpt(t, y, reset_lvl, dt, P)
        down_cross_times[obs_counter] = t
    end
    obs_times, down_cross_times
end

macro τ_summary(f)
    τs, dc = eval(f)
    ϵs = dc .- τs
    println("------------------------------------------------")
    println("mean for ϵ: ", round(mean(ϵs), digits=2), ", confidence intv: (",
            round(mean(ϵs)-2.0*std(ϵs), digits=2), ", ",
            round(mean(ϵs)+2.0*std(ϵs), digits=2), ").")
    println("------------------------------------------------")
    τs, dc
end

macro format_τ(f)
    τs, dc = eval(f)
    println("reformatting data...")
    collect(zip(dc[1:end-1], τs[2:end]))
end

function save_τ_to_file(datasets, filename)
    num_datasets = length(datasets)
    num_lines = map(x->length(x), datasets)
    @assert length(unique(num_lines)) == 1
    num_lines = num_lines[1]

    open(filename, "w") do f
        for i in 1:num_datasets
            ending = (i == num_datasets) ? "\n" : ", "
            write(f, string("reset", i, ", up_crossing", i, ending))
        end

        for j in 1:num_lines
            for i in 1:num_datasets
                ending = (i == num_datasets) ? "\n" : ", "
                write(f, string(datasets[i][j][1], ", ", datasets[i][j][2], ending))
            end
        end
    end
    print("Successfully written to a file ", filename, ".")
end


y0 = @SVector [0.0, 0.3, 0.1, 0.5]#[0.0, 0.5, 0.5, 0.06]
sim_parameters = (
    t0 = 0.0,
    y0 = y0,
    reset_lvl = -9.9,
    threshold = 10.0,
    dt = 0.001,
    P = HodgkinHuxley(parameters...),
    num_obs = 30,
)


Random.seed!(4)
current(t, P::HodgkinHuxley) = 8.0
data₁ = @format_τ @τ_summary run_experiment(sim_parameters...)

current(t, P::HodgkinHuxley) = 6.0
data₂ = @format_τ @τ_summary run_experiment(sim_parameters...)

current(t, P::HodgkinHuxley) = 4.0
data₃ = @format_τ @τ_summary run_experiment(sim_parameters...)
₁
current(t, P::HodgkinHuxley) = 2.0
data₄ = @format_τ @τ_summary run_experiment(sim_parameters...)


save_τ_to_file([data₁, data₂, data₃, data₄],
               joinpath(OUT_DIR, "first_passage_times_hodgkin_huxley.csv"))

#------------------------------------------------------------------------------#
#                       Some auxiliary, summary results
#------------------------------------------------------------------------------#
using KernelDensity

function plot_τ_hist(τs, ax = nothing)
    if ax === nothing
        fig, ax = plt.subplots()
    end
    ax.hist(τs, bins=300, normed=1)
    kde_τ = kde(τs)
    ax.plot(kde_τ.x, kde_τ.density)
    plt.tight_layout()
    ax
end

sim_parameters = (
    t0 = 0.0,
    y0 = y0,
    reset_lvl = -9.9,
    threshold = 10.0,
    dt = 0.01,
    P = HodgkinHuxley(parameters...),
    num_obs = 100000,
)

Random.seed!(4)
current(t, P::HodgkinHuxley) = 8.0
τ₁ = map(x->x[2]-x[1], @format_τ @τ_summary run_experiment(sim_parameters...))

current(t, P::HodgkinHuxley) = 6.0
τ₂ = map(x->x[2]-x[1], @format_τ @τ_summary run_experiment(sim_parameters...))

current(t, P::HodgkinHuxley) = 4.0
τ₃ = map(x->x[2]-x[1], @format_τ @τ_summary run_experiment(sim_parameters...))

current(t, P::HodgkinHuxley) = 2.0
τ₄ = map(x->x[2]-x[1], @format_τ @τ_summary run_experiment(sim_parameters...))


fig, ax = plt.subplots(2, 2, figsize=(15, 15))
plot_τ_hist(τ₁, ax[1,1])
plot_τ_hist(τ₂, ax[1,2])
plot_τ_hist(τ₃, ax[2,1])
plot_τ_hist(τ₄, ax[2,2])
