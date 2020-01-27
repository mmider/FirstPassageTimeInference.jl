using Distributions
import Random: rand!, rand
import Distributions: logpdf

struct CIRRandomWalk <: TunableTransitionKernel
    ϵ::Vector{Float64}
    function CIRRandomWalk(ϵ::Vector{Float64})
        @assert length(ϵ) == 4
        # step sizes for updates of (θ, α, low, σ)
        new(ϵ)
    end
end

rand!(rw::CIRRandomWalk, θ, i::Integer) = rand!(rw, θ, Val{i}())

function rand(rw::CIRRandomWalk, θ, i::Integer)
    θc = copy(θ)
    rand!(rw, θc, Val{i}())
end

function rand!(rw::CIRRandomWalk, θ, ::Val{1})
    θ[1] *= exp(rand(Uniform(-rw.ϵ[1], rw.ϵ[1])))
    θ
end

function rand!(rw::CIRRandomWalk, θ, ::Val{2})
    θ[2] += rand(Uniform(-min(rw.ϵ[2], θ[2]-0.75*θ[4]^2), rw.ϵ[2]))
    θ
end

function rand!(rw::CIRRandomWalk, θ, ::Val{3})
    θ[3] += rand(Uniform(-rw.ϵ[3], rw.ϵ[3]))
    θ
end

function rand!(rw::CIRRandomWalk, θ, ::Val{4})
    θ[4] *= exp(rand(Uniform(-rw.ϵ[4], min(rw.ϵ[4], log(2.0*sqrt(θ[2]/3.0)/θ[4])))))
    θ
end


function rand!(rw::CIRRandomWalk, θ)
    rand!(rw, θ, Val{1}())
    rand!(rw, θ, Val{2}())
    rand!(rw, θ, Val{3}())
    rand!(rw, θ, Val{4}())
    θ
end

function rand(rw::CIRRandomWalk, θ)
    θc = copy(θ)
    rand!(rw, θc)
end

function logpdf(rw::CIRRandomWalk, θ, θᵒ)
    c₁ = θ[1] != θᵒ[1] ? -log(θᵒ[1]) - log(2*rw.ϵ[1]) : 0.0
    c₂ = θ[2] != θᵒ[2] ? -log(rw.ϵ[2] + min(rw.ϵ[2], θ[2]-0.75*θ[4]^2)) : 0.0
    c₃ = θ[3] != θᵒ[3] ? -log(2*rw.ϵ[3]) : 0.0
    c₄ = θ[4] != θᵒ[4] ? -log(θᵒ[4]) - log(rw.ϵ[4] + min(rw.ϵ[4], log(2.0*sqrt(θ[2]/3.0)/θ[4]))) : 0.0
    c₁ + c₂ + c₃ + c₄
end
