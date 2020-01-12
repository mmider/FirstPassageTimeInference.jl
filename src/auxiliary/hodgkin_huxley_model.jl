using StaticArrays
using LinearAlgebra


#------------------------------------------------------------------------------#
#                   Definition of the Hodkin-Huxley model
#------------------------------------------------------------------------------#

struct HodgkinHuxley
    c_m::Float64
    g_K::Float64
    g_Na::Float64
    g_l::Float64
    v_K::Float64
    v_Na::Float64
    v_l::Float64
    σ::SArray{Tuple{4,4},Float64,2,16}
end

function b(t, y::SArray{Tuple{4},Float64,1,4}, P::HodgkinHuxley)
    @SVector[(current(t, P)
              - P.g_K * y[2]^4*(y[1]-P.v_K)
              - P.g_Na * y[3]^3*y[4]*(y[1]-P.v_Na)
              - P.g_l * (y[1] - P.v_l))/P.c_m,
              α_n(y[1]) * (1-y[2]) - β_n(y[1]) * y[2],
              α_m(y[1]) * (1-y[3]) - β_m(y[1]) * y[3],
              α_h(y[1]) * (1-y[4]) - β_h(y[1]) * y[4]]
end

σ(t, y::SArray{Tuple{4},Float64,1,4}, P::HodgkinHuxley) = P.σ

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


#------------------------------------------------------------------------------#
#                         Exploratory visualisations
#------------------------------------------------------------------------------#

function _dirty_simulate(y0, tt, P)
    N = length(tt)
    noise = randn(typeof(y0), N-1)
    XX = zeros(typeof(y0), N)
    XX[1] = y0
    for i in 1:N-1
        x = XX[i]
        dt = tt[i+1] - tt[i]
        XX[i+1] = x + b(tt[i], x, P) * dt + sqrt(dt) * σ(tt[i], x, P) * noise[i]
    end
    XX
end


parameters = (
    c_m = 1.0,
    g_K = 36.0,
    g_Na = 120.0,
    g_l = 0.3,
    v_K = -12.0,
    v_Na = 115.0,
    v_l = 10.613,
    σ = @SMatrix [ 0.001  0.0  0.0  0.0;
                    0.0 0.01  0.0  0.0;
                    0.0  0.0 0.01  0.0;
                    0.0  0.0  0.0 0.01;]
)

current(t, P::HodgkinHuxley) = 2.0

P_HH = HodgkinHuxley(parameters...)

tt = 0.0:0.001:500.0
y0 = @SVector [0.0, 0.3, 0.1, 0.5]#[0.0, 0.5, 0.5, 0.06]
XX = _dirty_simulate(y0, tt, P_HH)

using PyPlot

fig, ax = plt.subplots(4,1, figsize=(15,10))
for i in 1:4
    ax[i].plot(tt, map(x->x[i], XX))
end
ax[1].plot([0, 500], [-9.9,-9.9], linestyle="dashed", color="red")
ax[1].plot([0, 500], [10, 10], linestyle="dashed", color="green")

plt.tight_layout()




#------------------------------------------------------------------------------#
#                               the main experiment
#------------------------------------------------------------------------------#

function _dirty_simulate_fpt(t0, x0, xT, dt, P)
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

function _dirty_reset_fpt(t0, x0, xT, dt, P)
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
        y, t = _dirty_simulate_fpt(t, y, threshold, dt, P)
        obs_counter += 1
        obs_times[obs_counter] = t
        y, t = _dirty_reset_fpt(t, y, reset_lvl, dt, P)
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

current(t, P::HodgkinHuxley) = 2.0
data₄ = @format_τ @τ_summary run_experiment(sim_parameters...)


save_τ_to_file([data₁, data₂, data₃, data₄],
               joinpath(OUT_DIR, "first_passage_times_hodgkin_huxley.csv"))
