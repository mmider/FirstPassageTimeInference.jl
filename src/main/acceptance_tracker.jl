mutable struct AccptTracker
    num_prop::Int64
    num_accpt::Int64
    AccptTracker() = new(0, 0)
end

function register_accpt!(ac::AccptTracker, accepted)
    ac.num_prop += 1
    ac.num_accpt += 1*accepted
end

function accpt_rate(ac::AccptTracker)
    ac.num_prop == 0 ? 0.0 : ac.num_accpt / ac.num_prop
end

function reset!(ac::AccptTracker)
    ac.num_prop = 0.0
    ac.num_accpt = 0.0
end

sigmoid(x, a=1.0) = 1.0 / (1.0 + exp(-a*x))
logit(x, a=1.0) = (log(x) - log(1-x))/a

function adjust_ϵ(ϵ_old, p, a_r, δ, flip=1.0, f=identity, finv=identity)
    ϵ = finv(f(ϵ_old) + flip*(2*(a_r > p.trgt)-1)*δ)
    ϵ = max(min(ϵ,  p.max), p.min)    # trim excessive updates
end

compute_δ(p, mcmc_iter) = p.scale/sqrt(max(1.0, mcmc_iter/p.step-p.offset))
