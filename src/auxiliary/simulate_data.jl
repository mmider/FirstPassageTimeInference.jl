using Random

sampleFPT(t0, x0, xT, dt, P) = sampleFPT(t0, x0, xT, dt, P, state_space(P))

function sampleFPT(t0, x0, xT, dt, P, ::Unrestricted)
    y = x0
    t = t0
    sqdt = √dt
    while y < xT
        y += b(t, y, P)*dt + σ(t, y, P)*sqdt*randn(Float64)
        t += dt
    end
    t
end

function sampleFPT(t0, x0, xT, dt, P, ::MustBeAbove)
    y = x0
    t = t0
    sqdt = √dt
    while y < xT
        y_new = -Inf
        while y_new <= P.low
            y_new = y + b(t, y, P)*dt + σ(t, y, P)*sqdt*randn(Float64)
        end
        y = y_new
        t += dt
    end
    t
end

function run_experiment(l, L, dt, P, N, ϵ=0.0)
    samples = [(0.0,0.0) for _ in 1:N]
    t0 = 0.0
    for i in 1:N
        T = sampleFPT(t0, l, L, dt, P)
        samples[i] = (t0, T)
        t0 = T+ϵ
    end
    samples
end

flatten(x::Vector{<:Vector}) = collect(Iterators.flatten(x))
