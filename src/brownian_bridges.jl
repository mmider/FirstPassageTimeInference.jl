function sampleBB!(WW, Wnr)
    sample!(WW, Wnr)
    N = length(WW)
    yT = WW.yy[end]
    T = WW.tt[end]
    t0 = WW.tt[1]
    for i in 1:N
        WW.yy[i] -= yT * (WW.tt[i] - t0)/(T-t0)
    end
end
