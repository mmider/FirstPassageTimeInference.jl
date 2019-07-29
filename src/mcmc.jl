
struct Workspace{TW,TX}
    Wnr::Wiener{ℝ{3,Float64}}
    WW::Vector{TW}
    WWᵒ::Vector{TW}
    XX::Vector{TX}
    XXᵒ::Vector{TX}
    repo::Vector{Reposit}
    repoᵒ::Vector{Reposit}
    N::Int64

    function Workspace(dt, obsTimes, obsVals, P)
        Wnr = Wiener{ℝ{3,Float64}}()

        TW = typeof(sample([0], Wnr))
        TX = typeof(SamplePath([], zeros(Float64, 0)))

        m = length(obsTimes)
        WWᵒ = Vector{TW}(undef, m)
        XXᵒ = Vector{TX}(undef, m)
        repoᵒ = Vector{Reposit}(undef, m)
        for i in 1:m
            tt = timeGrid(obsTimes[i], dt)
            WWᵒ[i] = Bridge.samplepath(tt, zero(ℝ{3,Float64}))
            sampleBB!(WWᵒ[i], Wnr)
            XXᵒ[i] = Bridge.samplepath(tt, zero(Float64))
            repoᵒ[i] = Reposit(obsTimes[i]..., η.(obsVals[i], [P,P])...)
            Bessel!(Val{true}(), WWᵒ[i], XXᵒ[i], repoᵒ[i])
        end
        WW = deepcopy(WWᵒ)
        XX = deepcopy(XXᵒ)
        repo = deepcopy(repoᵒ)
        new{TW,TX}(Wnr, WW, WWᵒ, XX, XXᵒ, repo, repoᵒ, m)
    end
end

function timeGrid((t0,T), dt)
    N = length(t0:dt:T)
    dt = (T-t0)/(N-1)
    t0:dt:T
end

crankNicolson!(yᵒ, y, ρ) = (yᵒ .= √(1-ρ)*yᵒ + √(ρ)*y)

function impute!(ws::Workspace, P, ρ)
    for i in 1:ws.N
        sampleBB!(ws.WWᵒ[i], ws.Wnr)
        crankNicolson(ws.WWᵒ[i].yy, ws.WW[i].yy, ρ)
        Bessel!(Val{true}(), WWᵒ[i], XXᵒ[i], repoᵒ[i])
    end
end

function updateParams!(ws::Workspace, P)
end

function mcmc(obsTimes, obsVals, P, dt, numMCMCsteps, ρ)
    ws = Workspace(dt, obsTimes, obsVals, P)

    θs = Vector{typeof(params(P))}(undef, numMCMCsteps+1)
    θs[1] = params(P)
    for i in 1:numMCMCsteps
        impute!(ws, P)
        updateParams!(ws, P)
        θs[i+1] = params(P)
    end
    θs
end
