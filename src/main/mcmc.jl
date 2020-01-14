using Distributions

abstract type UpdateType end

struct MetropolisHastings <: UpdateType end
struct ConjugateUpdate <: UpdateType end

abstract type StateSpaceType end
struct MustBePositive <: StateSpaceType end
struct Unrestricted <: StateSpaceType end
struct MustBeAbove <: StateSpaceType
    low_bd::Float64
end

struct Workspace{TW,TX}
    Wnr::Wiener{ℝ{3,Float64}}
    WW::Vector{TW}
    WWᵒ::Vector{TW}
    XX::Vector{TX}
    XXᵒ::Vector{TX}
    repo::Vector{Reposit}
    repoᵒ::Vector{Reposit}
    ll::Vector{Float64}
    llᵒ::Vector{Float64}
    numAccpt::Vector{Int64}
    N::Int64

    function Workspace(dt, obsTimes, obsVals, P)
        Wnr = Wiener{ℝ{3,Float64}}()

        TW = typeof(sample([0], Wnr))
        TX = typeof(SamplePath([], zeros(Float64, 0)))

        m = length(obsTimes)
        WWᵒ = Vector{TW}(undef, m)
        XXᵒ = Vector{TX}(undef, m)
        repoᵒ = Vector{Reposit}(undef, m)
        llᵒ = zeros(Float64, m)

        for i in 1:m
            tt = timeGrid(obsTimes[i], dt)
            llᵒ[i] = -Inf
            while llᵒ[i] == -Inf
                WWᵒ[i] = Bridge.samplepath(tt, zero(ℝ{3,Float64}))
                sampleBB!(WWᵒ[i], Wnr)
                XXᵒ[i] = Bridge.samplepath(tt, zero(Float64))
                repoᵒ[i] = Reposit(obsTimes[i]..., η.(obsTimes[i], obsVals[i], [P,P])...)
                Bessel!(Val{true}(), WWᵒ[i], XXᵒ[i], repoᵒ[i])
                llᵒ[i] = pathLogLikhd(XXᵒ[i], P)
            end
        end
        WW = deepcopy(WWᵒ)
        XX = deepcopy(XXᵒ)
        repo = deepcopy(repoᵒ)
        ll = deepcopy(llᵒ)
        numAccpt = zeros(Float64, m)

        new{TW,TX}(Wnr, WW, WWᵒ, XX, XXᵒ, repo, repoᵒ, ll, llᵒ, numAccpt, m)
    end
end

function timeGrid((t0,T), dt)
    N = length(t0:dt:T)
    dt = (T-t0)/(N-1)
    t0:dt:T
end

function pathLogLikhd(XX, P)
    outside_state_space(XX, state_space(P)) && return -Inf
    N = length(XX)
    ll = 0.0
    for i in 1:N-1
        ll -= ϕ(XX.tt[i], XX.yy[i], P)*(XX.tt[i+1]-XX.tt[i])
    end
    ll
end

outside_state_space(XX, ::T) where T <: Unrestricted = false
outside_state_space(XX, ::T) where T <: MustBePositive = any(XX.yy .<= 0.0)
outside_state_space(XX, cond::T) where T <: MustBeAbove = any(XX.yy .< cond.low_bd)

crankNicolson!(yᵒ, y, ρ) = (yᵒ .= √(1-ρ)*yᵒ + √(ρ)*y)

function swap!(ws::Workspace, i)
    ws.XX[i], ws.XXᵒ[i] = ws.XXᵒ[i], ws.XX[i]
    ws.ll[i], ws.llᵒ[i] = ws.llᵒ[i], ws.ll[i]
end

function swap!(ws::Workspace)
    for i in 1:ws.N
        swap!(ws, i)
    end
end

function swapNoise!(ws::Workspace, i)
    ws.WW[i], ws.WWᵒ[i] = ws.WWᵒ[i], ws.WW[i]
end

function swapRepo!(ws::Workspace, i)
    ws.repoᵒ[i], ws.repo[i] = ws.repo[i], ws.repoᵒ[i]
end

function swapRepo!(ws::Workspace)
    for i in 1:ws.N
        swapRepo!(ws, i)
    end
end

function swap_likhd!(ws::Workspace)
    for i in 1:ws.N
        ws.ll[i], ws.llᵒ[i] = ws.llᵒ[i], ws.ll[i]
    end
end

function impute!(ws::Workspace, P, ρ)
    for i in 1:ws.N
        sampleBB!(ws.WWᵒ[i], ws.Wnr)
        crankNicolson!(ws.WWᵒ[i].yy, ws.WW[i].yy, ρ)
        Bessel!(Val{true}(), ws.WWᵒ[i], ws.XXᵒ[i], ws.repo[i])
        ws.llᵒ[i] = pathLogLikhd(ws.XXᵒ[i], P)
        if rand(Exponential()) ≥ -(ws.llᵒ[i]-ws.ll[i])
            swap!(ws, i)
            swapNoise!(ws, i)
            ws.numAccpt[i] += 1
        end
    end
end

fptlogpdf(x, t) = log(abs(x)) - 0.5*log(2.0*π) - 1.5*log(t) - 0.5*x^2/t


function obsLogLikhd(obs, t0::Float64, T::Float64, P)
    L = η(T, obs[2],  P)
    l = η(t0, obs[1], P)
    A(T, L, P) - A(t0, l, P) + fptlogpdf(L-l, T-t0)
end

function obsLogLikhd(obs, obsTimes::Vector{T}, P) where T
    N = length(obs)
    ll = 0.0
    for i in 1:N
        ll += obsLogLikhd(obs[i], obsTimes[i][1], obsTimes[i][2], P)
    end
    ll
end

function updateParams!(::MetropolisHastings, ws::Workspace, P, θ, tKernel, idx,
                       prior, obs, obsTimes, verbose, it)
    θᵒ = rand(tKernel, θ, idx)
    Pᵒ = clone(P, θᵒ)
    for i in 1:ws.N
        ws.repoᵒ[i] = Reposit(ws.repo[i], η.(obsTimes[i], obs[i], [Pᵒ,Pᵒ])...)
        Bessel!(Val{true}(), ws.WW[i], ws.XXᵒ[i], ws.repoᵒ[i])
        ws.llᵒ[i] = pathLogLikhd(ws.XXᵒ[i], Pᵒ)
    end
    llᵒ = sum(ws.llᵒ) + obsLogLikhd(obs, obsTimes, Pᵒ)
    ll = sum(ws.ll) + obsLogLikhd(obs, obsTimes, P)
    llr = ( llᵒ - ll + logpdf(prior, θᵒ[idx]) - logpdf(prior, θ[idx])
           + logpdf(tKernel, θᵒ, θ) - logpdf(tKernel, θ, θᵒ) )
    verbose && print("it: ", it, ", llᵒ: ", round(llᵒ, digits=3), ", ll: ",
                     round(ll, digits=3), ", llr: ", round(llr, digits=3), "\n")
    if rand(Exponential()) ≥ -llr
        swap!(ws)
        swapRepo!(ws)
        return true, θᵒ, Pᵒ
    else
        return false, θ, P
    end
end

function updateParams!(::ConjugateUpdate, ws::Workspace, P, θ, tKernel, idx,
                       prior, obs, obsTimes, verbose, it)
    θᵒ = conjugateDraw(θ, ws.XX, P, prior)
    Pᵒ = clone(P, θᵒ)
    for i in 1:ws.N
        ws.llᵒ[i] = pathLogLikhd(ws.XX[i], Pᵒ)
    end
    sum(ws.llᵒ) == -Inf && return false, θ, P # sample outside of support

    swap_likhd!(ws)
    ll = sum(ws.ll) + obsLogLikhd(obs, obsTimes, Pᵒ)

    verbose && print("it: ", it, ", ll: ", round(ll, digits=3), "\n")
    true, θᵒ, Pᵒ
end

function savePath!(::Val{true}, paths, XX, P, idx)
    xx = deepcopy(XX)
    for i in 1:length(xx)
        for j in 1:length(xx[i])
            xx[i].yy[j] = η⁻¹(xx[i].tt[j], xx[i].yy[j], P)
        end
    end
    paths[idx] = xx
end

savePath!(::Val{false}, ::Any, ::Any, ::Any, ::Any) = nothing

function mcmc(obsTimes, obsVals, P, dt, numMCMCsteps, ρ, updtParamIdx, tKernel,
              priors, updateType, saveIter, verbIter)
    ws = Workspace(dt, obsTimes, obsVals, P)

    θs = Vector{typeof(params(P))}(undef, numMCMCsteps+1)
    θ = θs[1] = params(P)
    paths = Vector{Any}(undef, div(numMCMCsteps, saveIter))

    num_param_updt = length(updtParamIdx)

    numProp = zeros(Int64, num_param_updt)
    numAccepted = zeros(Int64, num_param_updt)
    for i in 1:numMCMCsteps
        savePath!(Val{i % saveIter == 0}(), paths, ws.XX, P, div(i,saveIter))
        impute!(ws, P, ρ)
        if num_param_updt > 0
            idx₁ = mod1(i, num_param_updt)
            idx = updtParamIdx[idx₁]
            accepted, θ, P = updateParams!(updateType[idx], ws, P, θ, tKernel,
                                           idx, priors[idx], obsVals, obsTimes,
                                           i % verbIter == 0, i)
            numAccepted[idx₁] += accepted
            numProp[idx₁] += 1
        end
        θs[i+1] = copy(θ)
    end
    print("Acceptance rates for imputation: ", ws.numAccpt./numMCMCsteps, "\n")
    print("Acceptance rates for parameter update: ", numAccepted./numProp, "\n")
    θs, paths
end
