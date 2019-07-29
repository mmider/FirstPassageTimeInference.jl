
using Distributions

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
        ll = zeros(Float64, m)
        llᵒ = zeros(Float64, m)

        for i in 1:m
            tt = timeGrid(obsTimes[i], dt)
            WWᵒ[i] = Bridge.samplepath(tt, zero(ℝ{3,Float64}))
            sampleBB!(WWᵒ[i], Wnr)
            XXᵒ[i] = Bridge.samplepath(tt, zero(Float64))
            repoᵒ[i] = Reposit(obsTimes[i]..., η.(obsVals[i], [P,P])...)
            Bessel!(Val{true}(), WWᵒ[i], XXᵒ[i], repoᵒ[i])
            llᵒ[i] = ll[i] = pathLogLikhd(XXᵒ[i], P)
        end
        WW = deepcopy(WWᵒ)
        XX = deepcopy(XXᵒ)
        repo = deepcopy(repoᵒ)
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
    integrate(x -> -ϕ(x,P), XX)
end


crankNicolson!(yᵒ, y, ρ) = (yᵒ .= √(1-ρ)*yᵒ + √(ρ)*y)


function swap!(ws::Workspace, i)
    ws.WW[i], ws.WWᵒ[i] = ws.WWᵒ[i], ws.WW[i]
    ws.XX[i], ws.XXᵒ[i] = ws.XXᵒ[i], ws.XX[i]
    ws.ll[i], ws.llᵒ[i] = ws.llᵒ[i], ws.ll[i]
end

function swap!(ws::Workspace)
    for i in 1:ws.N
        swap!(ws, i)
    end
end

function impute!(ws::Workspace, P, ρ)
    for i in 1:ws.N
        sampleBB!(ws.WWᵒ[i], ws.Wnr)
        crankNicolson!(ws.WWᵒ[i].yy, ws.WW[i].yy, ρ)
        Bessel!(Val{true}(), ws.WWᵒ[i], ws.XXᵒ[i], ws.repo[i])
        ws.llᵒ[i] = pathLogLikhd(ws.XXᵒ[i], P)
        if rand(Exponential()) ≥ (ws.llᵒ[i]-ws.ll[i])
            swap!(ws, i)
            ws.numAccpt[i] += 1
        end
    end
end

fptlogpdf(x, t) = log(abs(x)) - 0.5*log(2.0*π*t^3) - 0.5*x^2/t


function obsLogLikhd(obs, t::Float64, P)
    L = η(obs[2], P)
    l = η(obs[1], P)
    A(L, P) - A(l, P) + logD(obs[1], P) + fptlogpdf(L-l, t)
end

function obsLogLikhd(obs, obsTimes::Vector{T}, P) where T
    print("from here...\n")
    N = length(obs)
    ll = 0
    for i in 1:N
        t = obsTimes[i][2] - obsTimes[i][1]
        ll += obsLogLikhd(obs[i], t, P)
    end
    ll
end

function updateParams!(ws::Workspace, P, θ, tKernel, idx, prior, obs, obsTimes)
    θᵒ = rand(tKernel, θ, idx)
    Pᵒ = clone(P, θᵒ)
    for i in 1:ws.N
        ws.repoᵒ[i] = Reposit(ws.repo[i], η.(obs[i], [Pᵒ,Pᵒ])...)
        Bessel!(Val{true}(), ws.WW[i], ws.XXᵒ[i], ws.repoᵒ[i])
        ws.llᵒ[i] = pathLogLikhd(ws.XXᵒ[i], Pᵒ)
    end
    llr = ( sum(ws.llᵒ) - sum(ws.ll)
           + obsLogLikhd(obs, obsTimes, Pᵒ) - obsLogLikhd(obs, obsTimes, P)
           + logpdf(prior, θᵒ[idx]) - logpdf(prior, θ[idx])
           + logpdf(tKernel, θᵒ, θ) - logpdf(tKernel, θ, θᵒ) )
    if rand(Exponential()) ≥ llr
        swap!(ws)
        return true, θᵒ, Pᵒ
    else
        return false, θ, P
    end
end

function mcmc(obsTimes, obsVals, P, dt, numMCMCsteps, ρ, updtParamIdx, tKernel,
              priors, saveIter, verbIter)
    ws = Workspace(dt, obsTimes, obsVals, P)

    θs = Vector{typeof(params(P))}(undef, numMCMCsteps+1)
    θ = θs[1] = params(P)
    paths = Vector{Any}(undef, div(numMCMCsteps, saveIter))

    N = length(updtParamIdx)

    numAccepted = 0
    for i in 1:numMCMCsteps
        (i % verbIter == 0) && print("Iteration ", i, "\n")
        (i % saveIter == 0) && (paths[div(i,saveIter)] = deepcopy(ws.XX))
        impute!(ws, P, ρ)
        if N > 0
            idx = mod1(i, N)
            accepted, θ, P = updateParams!(ws, P, θ, tKernel, idx, priors[idx],
                                           obsVals, obsTimes)
            numAccepted += accepted
        end
        θs[i+1] = θ
    end
    print("Acceptance rates for imputation: ", ws.numAccpt./numMCMCsteps, "\n")
    print("Acceptance rates for parameter update: ", numAccepted/numMCMCsteps, "\n")
    θs, paths
end
