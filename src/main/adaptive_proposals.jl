const _DEFAULT_IMP_P = (
    scale = 0.1,
    step = 10.0,
    offset = 20,
    trgt = 0.234,
    min = 0.0,
    max = 1.0-1e-7,
    num_ρs_to_display=5,
)

const _DEFAULT_UPDT_P = (
    scale = 0.1,
    step = 10.0,
    offset = 20,
    trgt = 0.234,
    min = 1e-7,
    max = 99999.9,
)


time_to_update_ρs(i) = (i % 100 == 0) && (i > 10)
time_to_update_tkern(i, N) = (div(i, N) % 100 == 0) && (div(i, N) > 10)

function update_ρs!(ws::Workspace, mcmc_iter, p)
    m = length(ws.ρs)
    δ = compute_δ(p, mcmc_iter)
    for i in 1:m
        a_r = accpt_rate(ws.imp_accpt[i])
        ws.ρs[i] = adjust_ϵ(ws.ρs[i], p, a_r, δ, -1.0, logit, sigmoid)
        reset!(ws.imp_accpt[i])
    end
end

function display_ρ_status(ws::Workspace, max_num_to_display=5)
    println("* * *")
    println("Updating ρs...")
    m = length(ws.ρs)
    num_not_displayed = m - 2*max_num_to_display
    override_and_display_all = num_not_displayed <= 3
    for i in 1:m
        if ( override_and_display_all ||
             (i <= max_num_to_display) ||
             (i > m-max_num_to_display) )
             a_r = round(accpt_rate(ws.imp_accpt[i]), digits=3)
             ρ = round(ws.ρs[i], digits=3)
             print("[$i], ar: $a_r, ρ: $ρ; ")
         end
         if i == max_num_to_display && !override_and_display_all
             print("... ($num_not_displayed ρs are not displayed) ...")
         end
    end
    println("\n------------")
end

function display_ρs_only(ws::Workspace, max_num_to_display=5)
    println("New ρs: ")
    m = length(ws.ρs)
    num_not_displayed = m - 2*max_num_to_display
    override_and_display_all = num_not_displayed <= 3
    for i in 1:m
        if ( override_and_display_all ||
             (i <= max_num_to_display) ||
             (i > m-max_num_to_display) )
            ρ = round(ws.ρs[i], digits=3)
            print("[$i], ρ: $ρ; ")
        end
        if i == max_num_to_display && !override_and_display_all
            print("... ($num_not_displayed ρs are not displayed) ...")
        end
    end
    println("\n------------\n")
end

update_tkern!(ws::Workspace, ::Any) = nothing

function update_tkern!(ws::Workspace, t_kernel::T, mcmc_iter, p
                       ) where {T <: TunableTransitionKernel}
    m = length(ws.updt_accpt)
    δ = compute_δ(p, mcmc_iter)
    for i in 1:m
        a_r = accpt_rate(ws.updt_accpt[i])
        t_kernel.ϵ[i] = adjust_ϵ(t_kernel.ϵ[i], p, a_r, δ)
    end
end

function display_kernel_status(ws::Workspace, t_kernel::T
                               ) where {T <: TunableTransitionKernel}
    m = length(ws.updt_accpt)
    println("* * *")
    println("Updating transition kernels...")
    for i in 1:m
        a_r = round(accpt_rate(ws.updt_accpt[i]), digits=3)
        ϵ = round(t_kernel.ϵ[i], digits=3)
        print("[$i], ar: $a_r, ϵ: $ϵ; ")
    end
    println("\n------------")
end

display_kernel_status(ws::Workspace, ::Any) = nothing

function display_kernel_only(t_kernel::T) where {T <: TunableTransitionKernel}
    println("New transition kernel: ")
    for (i,ϵ) in enumerate(map(x-> round(x, digits=3), t_kernel.ϵ))
        print("[$i], ϵ: $ϵ; ")
    end
    println("\n------------\n")
end

display_kernel_only(::Any) = nothing
