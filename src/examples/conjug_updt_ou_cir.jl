function conjugateDraw(θ, XX, P::T, prior) where T <: Union{OrnsteinUhlenbeck,CoxIngersollRoss}
    μ = @SVector[0.0, 0.0]
    𝓦 = μ*μ'
    ϑ = @SVector[θ[1], θ[2]]
    μ, 𝓦 = _conjugateDraw(μ, 𝓦, XX, P)

    Σ = inv(𝓦 + inv(Matrix(prior.Σ)))
    Σ = (Σ + Σ')/2
    μₚₒₛₜ = Σ * (μ + Vector(prior.Σ\prior.μ))

    ϑᵒ = rand(Gaussian(μₚₒₛₜ, Σ))
    θᵒ = copy(θ)
    θᵒ[1], θᵒ[2] = ϑᵒ[1], ϑᵒ[2]
    θᵒ
end

function _conjugateDraw(μ, 𝓦, XX, P::T) where T <: Union{OrnsteinUhlenbeck, CoxIngersollRoss}
    for X in XX
        for i in 1:length(X)-1
            φₜ = φ(X.tt[i], X.yy[i], P)
            φₜᶜ = φᶜ(X.tt[i], X.yy[i], P)
            dt = X.tt[i+1]-X.tt[i]
            dy = X.yy[i+1]-X.yy[i]
            μ = μ + φₜ*dy - φₜ*φₜᶜ*dt
            𝓦 = 𝓦 + φₜ*φₜ'*dt
        end
    end
    μ, 𝓦
end
