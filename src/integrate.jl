function integrate(f, XX::SamplePath)
    N = length(XX)
    v = 0.0
    for i in 1:N-1
        v += f(XX.yy[i])*(XX.tt[i+1]-XX.tt[i])
    end
    v
end
