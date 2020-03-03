#==============================================================================#
#
#       This file runs independently from the rest and is used to generate some
#       data for the experiments and visualise the Hodgkin-Huxley model to
#       better understand its dynamics
#
#==============================================================================#

using StaticArrays
using LinearAlgebra, Statistics
using Parameters
using Random
using PyPlot
const 𝕍{N} = SArray{Tuple{N},Float64,1,N} where N

OUT_DIR = joinpath(Base.source_dir(), "..", "..", "output")
include("hodgkin_huxley_model_supporting_functions.jl")
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

function synaptic_current(t, y::𝕍{6}, P::HodgkinHuxleySSI)
    ( y[5]*(P.v_E - y[1]) + y[6]*(P.v_I - y[1]) ) / P.membrane_area
end

function standard_HH_currents(t, y, P::T) where {T<:HodgkinHuxley}
    ( P.g_K * y[2]^4*(P.v_K-y[1])
     + P.g_Na * y[3]^3*y[4]*(P.v_Na-y[1])
     + P.g_l * (P.v_l-y[1]) )
end

drift(y::Float64, p::OU_params) = p.θ*(p.μ-y)

function drift(t, y::𝕍{6}, P::HodgkinHuxleySSI)
    @SVector[(synaptic_current(t, y, P) + standard_HH_currents(t, y, P))/P.c_m,
              α_n(y[1]) * (1-y[2]) - β_n(y[1]) * y[2],
              α_m(y[1]) * (1-y[3]) - β_m(y[1]) * y[3],
              α_h(y[1]) * (1-y[4]) - β_h(y[1]) * y[4],
              drift(y[5], P.E),
              drift(y[6], P.I)]
end

σ(t, y::𝕍{6}, P::HodgkinHuxleySSI) = @SMatrix[ 0.0 0.0;
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
    I = OU_params(1.0/8.0, 0.1125, 0.01),
)

artificial_excitatory = (
    E = OU_params(1.0/2.0, 0.11, 0.0125),
    I = OU_params(1.0/8.0, 0.0, 0.0),
)

#==============================================================================#
#
#                               Visualisations
#
#==============================================================================#

#NOTE functions `_simulate` and `quick_plot` are defined in a companion file
# `hodgkin_huxley_supporting_functions.jl`

# Pick out a type of the neuron for simulations
neuron = artificial#_excitatory

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
y0 = 𝕍{6}(resting_HH..., neuron.E.μ, neuron.I.μ)
XX = _simulate(𝕍{2}, y0, tt, P_HH)

ax = quick_plot(tt, XX; avg_synaptic_input=15.0)
# zoom-in to see if the avg synaptic input makes sense
ax[7].set_ylim([13.0, 16.0])
# zoom-in to see if the reset level makes sense
ax[1].set_ylim([-10.0, -8.2])
# zoom-in to see if the threshold level makes sense
ax[1].set_ylim([10.0, 13.0])

#------------------------------------------------------------------
#
#       Let's add a `Conductance vs membrane potential plot`
#
#------------------------------------------------------------------
Random.seed!(13)
fig, ax = plt.subplots(1, figsize=(10, 5))
samples = []
y0_fixed = copy(y0)
for i in 1:20
    println(i, "...")
    global y0, tt, P_HH
    XX = _simulate(𝕍{2}, y0, tt, P_HH)
    append!(samples, XX[1:100:end-1])
    y0 = copy(XX[end])
end
y0 = copy(y0_fixed)

ax.plot([4.5],[0.39], marker="o", markerfacecolor="black", markeredgecolor="black", markersize=10)
ax.plot(map(x->x[1], samples[474101:476000]), map(x->x[2], samples[474101:476000]), color="black", linestyle="dashed")
ax.plot(map(x->x[1], samples[492401:493900]), map(x->x[2], samples[492401:493900]), color="black", linestyle="dashed")
plt.tight_layout()
#=
for i in 1:20
    offset = i*1000
    println(i, "...")
    ax.plot(map(x->x[1], samples[offset+1:offset+1000]), map(x->x[2], samples[offset+1:offset+1000]), color="black")
    sleep(1)
end
=#

using Makie

scene = Scene()
time_node = Node(1)

x1 = map(x->x[1], samples)
x2 = map(x->x[2], samples)

foo(t::Float64) = foo(Int64(t))
foo(t::Int64) = (x1[t:t+100], x2[t:t+100])

scatter!(scene, [4.5], [0.39], markersize=1, color="red")
plot!(map(x->x[1], samples[474101:476000]), map(x->x[2], samples[474101:476000]), linestyle=:dash, color=:steelblue)
plot!(map(x->x[1], samples[492401:493900]), map(x->x[2], samples[492401:493900]), linestyle=:dash, color=:steelblue)
p = scatter!(scene, lift(t-> foo(t), time_node), limits = FRect(-5.0, 0.3, 100.0, 0.4), color=100:-1:1 ,colormap=:grays, markersize=1)[end]
N = 100000
record(scene, "output.mp4", 1:N; framerate=240) do i
    push!(time_node, i)
end

using Makie

scene = Scene()
f(t, v, s) = (sin(v + t) * s, cos(v + t) * s)
time_node = Node(0.0)
p1 = scatter!(scene, lift(t-> f.(t, range(0, stop = 2pi, length = 50), 1), time_node))[end]
p2 = scatter!(scene, lift(t-> f.(t * 2.0, range(0, stop = 2pi, length = 50), 1.5), time_node))[end]
points = lift(p1[1], p2[1]) do pos1, pos2
    map((a, b)-> (a, b), pos1, pos2)
end
linesegments!(scene, points)
N = 150
record(scene, "output.mp4", range(0, stop = 10, length = N)) do i
    push!(time_node, i)
end




#==============================================================================#
#
#                           THE MAIN EXPERIMENT
#
#==============================================================================#

#NOTE functions `run_experiment` and `save_τ_to_file` as well as macros
# `τ_summary`, `format_τ` are defined in a companion file
# `hodgkin_huxley_supporting_functions.jl`

Random.seed!(4)

# perform three experiments with various levels of mean excitatory input
data = map([0.08, 0.11, 0.14, 0.17]) do g_E
    neuron = (  # excitatory input only
        E = OU_params(1.0/2.0, g_E, 0.0125),
        I = OU_params(1.0/8.0, 0.0, 0.0),
    )
    P = HodgkinHuxleySSI(
        v_E = 75.0,
        v_I = 0.0,
        E = neuron.E,
        I = neuron.I
    )
    print( "\n***\nStarting the experiment with g_E = $g_E\n",
           "The 'average' synaptic input is: ",
           P.E.μ*(P.v_E-resting_HH[1]), "\n" )

    y0 = 𝕍{6}(resting_HH..., neuron.E.μ, neuron.I.μ)

    sim_parameters = (
        noise_type = 𝕍{2},
        t0 = 0.0,
        x0 = y0,
        reset_lvl = -8.3,
        threshold = 13.0,
        dt = 0.001,
        P = P,
        num_obs = 31,
    )
    @format_τ @τ_summary run_experiment(sim_parameters...)
end
# cache values of synaptic input: 6.0, 8.25, 10.5, 12.75

save_τ_to_file(data, joinpath(OUT_DIR, "first_passage_times_hodgkin_huxley.csv"))

#==============================================================================#
#
#                   Visualise first-passage time densities
#
#==============================================================================#

#NOTE a function `plot_many_τ_hist` is defined in a companion file
# `hodgkin_huxley_supporting_functions.jl`

saved_neuron = (  # for g_E in [0.08, 0.11, 0.14, 0.17]
    E = OU_params(1.0/2.0, 0.1 + g_E, 0.0225),
    I = OU_params(1.0/8.0, 0.1525, 0.02),
)

Random.seed!(4)

τs_HH = map([0.08, 0.11, 0.14, 0.17]) do g_E
    neuron = (  # excitatory input only
        E = OU_params(1.0/2.0, 0.1 + g_E, 0.0325),
        I = OU_params(1.0/8.0, 0.1525, 0.03),
    )
    P = HodgkinHuxleySSI(
        v_E = 75.0,
        v_I = 0.0,
        E = neuron.E,
        I = neuron.I
    )
    y0 = 𝕍{6}(resting_HH..., neuron.E.μ, neuron.I.μ)

    sim_parameters = (
        noise_type = 𝕍{2},
        t0 = 0.0,
        x0 = y0,
        reset_lvl = -7.5,
        threshold = 15.0,
        dt = 0.01,
        P = P,
        num_obs = Int64(1e5),
    )
    map(x->x[2]-x[1], @format_τ @τ_summary run_experiment(sim_parameters...))
end

axs = plot_many_τ_hist(τs_HH, nbins=600)
# an interesting zoom-in...
for ax in axs ax.set_ylim([0.0, 0.02]) end
for ax in axs ax.set_xlim([0, 100]) end
