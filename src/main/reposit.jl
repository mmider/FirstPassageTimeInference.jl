struct Reposit
    t0::Float64
    T::Float64
    x0::Float64
    xT::Float64
    Reposit(t0::Float64, T::Float64, x0::Float64, xT::Float64) = new(t0, T, x0, xT)
    Reposit(r::Reposit, x0::Float64, xT::Float64) = new(r.t0, r.T, x0, xT)
end

Φ(r::Reposit, x, t) = x + r.x0*(1 - (t-r.t0)/(r.T-r.t0)) + r.xT * (t-r.t0)/(r.T-r.t0)

function Φ!(r::Reposit, ww::Vector{Float64}, tt::Vector{Float64}, xx::Vector{Float64})
    N = length(ww)
    for i in 1:N
        xx[i] = Φ(r, xx[i], tt[i])
    end
end

Φ₁(r::Reposit, x, t) = x + (r.xT-r.x0) * (1.0 - (t-r.t0)/(r.T-r.t0))
Φ₂(r::Reposit, x) = -x + r.xT
