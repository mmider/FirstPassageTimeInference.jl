using KernelDensity

"""
    _simulate(noise_type::Type{T}, y0, tt, P) where T

Simulate a diffusion path on a time grid `tt`, starting from a state `y0`, for
a diffusion law `P` and using the driving Wiener noise of type `T`.
"""
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

"""
    quick_plot(tt, XX, skip=100; reset_lvl=-8.5, threshold=12.0,
               avg_synaptic_input=nothing)

Function used for quick visualisations of the simulated Hodgkin-Huxley model
"""
function quick_plot(tt, XX, skip=100; reset_lvl=-8.5, threshold=12.0,
                    avg_synaptic_input=nothing)
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
    if avg_synaptic_input !== nothing
        ax[7].plot([0, T], [avg_synaptic_input,avg_synaptic_input],
                   linestyle="dashed", color="orange")
    end

    plt.tight_layout()
    ax
end

"""
    crossing_condition_not_satisfied(y, threshold, ::Val{true})

Returns true only if an up-crossing has not been reached
"""
crossing_condition_not_satisfied(y, threshold, ::Val{true}) = y < threshold

"""
    crossing_condition_not_satisfied(y, threshold, ::Val{true})

Returns true only if a down-crossing has not been reached
"""
crossing_condition_not_satisfied(y, threshold, ::Val{false}) = y > threshold

"""
    _simulate_fpt(noise_type::Type{K}, t0, x0, xT, dt, P, up_crossing=true
                  ) where K

Simulate a single first-passage time of the first coordinate to level `xT` for a
diffusion with law `P`, which started from `x0`, using time-step `dt`.
"""
function _simulate_fpt(noise_type::Type{K}, t0, x0, xT, dt, P, up_crossing=true
                       ) where K
    y = x0
    t = t0
    sqdt = √dt
    crossing_type = Val{up_crossing}()
    while crossing_condition_not_satisfied(y[1], xT, crossing_type)
        y += drift(t, y, P) * dt + sqdt * σ(t, y, P) * randn(K)
        t += dt
    end
    y, t
end

"""
    run_experiment(noise_type::Type{K}, t0, x0, reset_lvl, threshold, dt, P,
                   num_obs) where K

Simulate multiple first passage times
"""
function run_experiment(noise_type::Type{K}, t0, x0, reset_lvl, threshold, dt,
                        P, num_obs) where K
    y =  x0
    t = t0
    obs_times = zeros(Float64, num_obs)
    down_cross_times = zeros(Float64, num_obs)
    obs_counter = 0
    while obs_counter < num_obs
        y, t = _simulate_fpt(noise_type, t, y, threshold, dt, P, true)
        obs_counter += 1
        obs_times[obs_counter] = t
        y, t = _simulate_fpt(noise_type, t, y, reset_lvl, dt, P, false)
        down_cross_times[obs_counter] = t
    end
    obs_times, down_cross_times
end

"""
    τ_summary(f)

Print-out some statistical summary about the distribution of the refractory
period---very useful for judging whether the chosen upper and lower thresholds
made sense.
"""
macro τ_summary(f)
    return quote
        τs, dc = $(esc(f))
        ϵs = dc .- τs
        println("------------------------------------------------")
        println("mean for ϵ: ", round(mean(ϵs), digits=2), ", confidence intv: (",
                round(mean(ϵs)-2.0*std(ϵs), digits=2), ", ",
                round(mean(ϵs)+2.0*std(ϵs), digits=2), ").")
        println("------------------------------------------------")
        τs, dc
    end
end

"""
    format_τ(f)

Reformat the simulated data to a format in which it can be saved
"""
macro format_τ(f)
    return quote
        τs, dc = $(esc(f))
        println("reformatting data...")
        collect(zip(dc[1:end-1], τs[2:end]))
    end
end

"""
    save_τ_to_file(datasets, filename)

Save first-passage time data to a file
"""
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

"""
    read_τ_data(filename)

Read first-passage time data from a file
"""
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


"""
    plot_τ_hist(τs, ax = nothing; nbins=300)

Plot a single histogram for first-passage times, add kernel density estimates
"""
function plot_τ_hist(τs, ax = nothing; nbins=300)
    if ax === nothing
        fig, ax = plt.subplots()
    end
    ax.hist(τs, bins=nbins, density=true)
    kde_τ = kde(τs)
    ax.plot(kde_τ.x, kde_τ.density)
    plt.tight_layout()
    ax
end

"""
    plot_many_τ_hist(τs; nbins=300, ax=nothing)

Plot multiple histograms for first-passage times, add kernel density estimates
"""
function plot_many_τ_hist(τs; nbins=300, ax=nothing)
    N = length(τs)
    if ax === nothing
        fig, ax = plt.subplots(1, N, figsize=(15,5))
    end
    for i in 1:N
        plot_τ_hist(τs[i], ax[i], nbins=nbins)
    end
    ax
end
