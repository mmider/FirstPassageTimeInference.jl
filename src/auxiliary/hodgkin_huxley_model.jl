using StaticArrays
using LinearAlgebra

struct HodgkinHuxley
    c_m::Float64
    g_K::Float64
    g_Na::Float64
    g_l::Float64
    v_K::Float64
    v_Na::Float64
    v_l::Float64
    σ::SArray{Tuple{4,4},Float64,2,16}
end

function b(t, y::SArray{Tuple{4},Float64,1,4}, P::HodgkinHuxley)
    @SVector[(current(t, P)
              - P.g_K * y[2]^4*(y[1]-P.v_K)
              - P.g_Na * y[3]^3*y[4]*(y[1]-P.v_Na)
              - P.g_l * (y[1] - P.v_l))/P.c_m,
              α_n(y[1]) * (1-y[2]) - β_n(y[1]) * y[2],
              α_m(y[1]) * (1-y[3]) - β_m(y[1]) * y[3],
              α_h(y[1]) * (1-y[4]) - β_h(y[1]) * y[4]]
end

σ(t, y::SArray{Tuple{4},Float64,1,4}, P::HodgkinHuxley) = P.σ

function α_n(y::Float64)
    x = 0.1*(10.0 - y)
    0.1*x/(exp(x)-1.0)
end

function α_m(y::Float64)
    x = 0.1*(25.0 - y)
    x/(exp(x)-1.0)
end

α_h(y::Float64) = 0.07*exp(-0.05*y)

β_n(y::Float64) = 0.125*exp(-0.0125*y)

β_m(y::Float64) = 4.0*exp(-y/18.0)

β_h(y::Float64) = 1.0/(exp(3.0-0.1*y)+1.0)

current(t, P::HodgkinHuxley) = 9


function _dirty_simulate(y0, tt, P)
    N = length(tt)
    noise = randn(typeof(y0), N-1)
    XX = zeros(typeof(y0), N)
    XX[1] = y0
    for i in 1:N-1
        x = XX[i]
        dt = tt[i+1] - tt[i]
        XX[i+1] = x + b(tt[i], x, P) * dt + sqrt(dt) * σ(tt[i], x, P) * noise[i]
    end
    XX
end

parameters = (
    c_m = 1.0,
    g_K = 36.0,
    g_Na = 120.0,
    g_l = 0.3,
    v_K = -12.0,
    v_Na = 115.0,
    v_l = 10.613,
    σ = @SMatrix [ 0.001  0.0  0.0  0.0;
                    0.0 0.01  0.0  0.0;
                    0.0  0.0 0.01  0.0;
                    0.0  0.0  0.0 0.01;]
)

P_HH = HodgkinHuxley(parameters...)

tt = 0.0:0.01:500.0
y0 = @SVector [0.0, 0.5, 0.5, 0.06]
XX = _dirty_simulate(y0, tt, P_HH)

using PyPlot

fig, ax = plt.subplots(4,1, figsize=(15,10))
for i in 1:4
    ax[i].plot(tt, map(x->x[i], XX))
end

plt.tight_layout()
