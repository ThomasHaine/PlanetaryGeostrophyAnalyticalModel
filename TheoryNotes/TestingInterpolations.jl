using Interpolations

# grids
xgrid = range(0, 360, length=181)   # longitude-like
ygrid = range(-80, 80, length=161)  # latitude-like
zgrid = range(-5000, 0, length=101) # depth

# fake data
Bdata = [sinpi(x/180) * cospi(y/180) * exp(z/5000) for x in xgrid, y in ygrid, z in zgrid]
bsdata = [sinpi(x/180) * cospi(y/180) for x in xgrid, y in ygrid]

# cubic spline interpolants
itpB = interpolate(Bdata, BSpline(Cubic(Line(OnGrid()))))
sitpB = scale(itpB, xgrid, ygrid, zgrid)
etpB = extrapolate(sitpB, Flat())

itpbs = interpolate(bsdata, BSpline(Cubic(Line(OnGrid()))))
sitpbs = scale(itpbs, xgrid, ygrid)
etpbs = extrapolate(sitpbs, Flat())

# evaluate
Bval = etpB(10.0, 20.0, -1000.0)
bsval = etpbs(10.0, 20.0)

println(Bval)
println(bsval)

dBdx = x -> Interpolations.gradient(sitpB, x, 20.0, -1000.0)[1]
dBdy = x -> Interpolations.gradient(sitpB, x, 20.0, -1000.0)[2]
dBdz = x -> Interpolations.gradient(sitpB, x, 20.0, -1000.0)[3]

dBdx(10.0)

parent(sitpB)