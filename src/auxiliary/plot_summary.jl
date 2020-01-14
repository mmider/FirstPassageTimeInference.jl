using PyPlot

function standard_summary_plot(P, paths, θs, θ_truth, offset=100, num_intv_to_plot=6)
    num_params = length(θs[1])
    fig, ax = plt.subplots(num_params+1,1, figsize=(20,10))
    plt.tight_layout()
    N, M = length(paths), length(paths[1])
    for i in max(1,N-offset):N
        for j in 1:min(M,num_intv_to_plot)
            ax[1].plot(paths[i][j].tt, paths[i][j].yy,
                       label="", alpha=0.2, color="steelblue", linewidth=0.4)
        end
    end

    for i in 1:num_params
        ax[i+1].plot([θ[i] for θ in θs])
        ax[i+1].plot([0.0, length(θs)], [θ_truth[i], θ_truth[i]], linestyle="dashed",
                     linewidth=3.0, color="orange")
    end
    ax
end


macro timeit(f)
      start = time()
      out = eval(f)
      elapsed = time() - start
      print("Time elapsed: ", elapsed, "\n")
      (output = out, time = elapsed)
end
