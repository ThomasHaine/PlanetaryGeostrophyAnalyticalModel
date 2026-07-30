using NCDatasets
using GibbsSeaWater
using Interpolations
using Statistics
using Random
using Plots

# ------------------------------
# Read climatological T, S data
# ------------------------------
"""
    read_woa_climatology(tempfile, saltfile;
                         tempvar="t_an", saltvar="s_an",
                         lonvar="lon", latvar="lat", zvar="depth")

Read WOA climatology temperature and salinity from separate files.
Returns x, y, z, T, S with arrays shaped (nx, ny, nz).
"""
function read_woa_climatology(tempfile, saltfile;
    tempvar="t_an", saltvar="s_an",
    lonvar="lon", latvar="lat", zvar="depth")

    print("Reading WOA climatology from: ",tempfile," and ",saltfile,"...")
    dT = NCDataset(tempfile)

    x = collect(dT[lonvar][:])
    y = collect(dT[latvar][:])
    z = collect(dT[zvar][:])

    Traw = Array(dT[tempvar][:, :, :, :])
    close(dT)

    dS = NCDataset(saltfile)
    Sraw = Array(dS[saltvar][:, :, :, :])
    close(dS)

    # WOA files are typically (lon, lat, depth, time)
    T = dropdims(Traw; dims=4)
    S = dropdims(Sraw; dims=4)

    # Replace missings with NaN and convert to Float64
    T = Float64.(coalesce.(T, NaN))
    S = Float64.(coalesce.(S, NaN))
    println("done.")

    return x, y, z, T, S
end

function inspect_nc_var(filename, varname)
    ds = NCDataset(filename)
    v = ds[varname]
    println("Variable: ", varname)
    println("Fieldnames: ", fieldnames(typeof(v)))
    println("Size: ", size(v))
    println("Eltype: ", eltype(v))
    close(ds)
end

function teos10_density(SP, t, p, lon, lat)
    SA = gsw_sa_from_sp(SP, p, lon, lat)
    CT = gsw_ct_from_t(SA, t, p)
    ρ  = gsw_rho(SA, CT, p)
    return ρ
end

function teos10_buoyancy(SP, t, p, lon, lat)
    ρ = teos10_density(SP, t, p, lon, lat)
    return -g * (ρ - ρ0) / ρ0
end

finite_mask(A) = isfinite.(A)

# ------------------------------
# Compute buoyancy fields
# ------------------------------
"""
    compute_buoyancy_fields(T, S)

Return 3D buoyancy B and 2D surface buoyancy bs.
Assumes T and S are size (nx, ny, nz).
Surface buoyancy is taken at the top level z = 1 (adjust if needed).
"""
function compute_buoyancy_TEOS10(x, y, z, T, S)
    print("Computing buoyancy using TEOS10 functions...")
    @assert size(T) == size(S)
    nx, ny, nz = size(T)

    B = fill(NaN, nx, ny, nz)

    for i in 1:nx, j in 1:ny, k in 1:nz
        if isfinite(T[i, j, k]) && isfinite(S[i, j, k])
            lon = x[i]
            lat = y[j]
            p   = gsw_p_from_z(z[k], lat)
            B[i, j, k] = teos10_buoyancy(S[i, j, k], T[i, j, k], p, lon, lat)
        end
    end

    bs = fill(NaN, nx, ny)
    for i in 1:nx, j in 1:ny
        if isfinite(T[i, j, 1]) && isfinite(S[i, j, 1])
            lon = x[i]
            lat = y[j]
            p   = gsw_p_from_z(z[1], lat)
            bs[i, j] = teos10_buoyancy(S[i, j, 1], T[i, j, 1], p, lon, lat)
        end
    end
    println("done.")
    return B, bs
end

function fill_nan_neighbors!(A; maxiter=100, radius=2)
    print("Filling NaNs in 3D...")
    A = copy(A)
    nx, ny, nz = size(A)

    for iter in 1:maxiter
        changed = false
        B = copy(A)

        for i in 1:nx, j in 1:ny, k in 1:nz
            if !isfinite(A[i,j,k])
                vals = Float64[]
                for di in -radius:radius, dj in -radius:radius, dk in -radius:radius
                    ii, jj, kk = i+di, j+dj, k+dk
                    if 1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz && isfinite(A[ii,jj,kk])
                        push!(vals, A[ii,jj,kk])
                    end
                end
                if !isempty(vals)
                    B[i,j,k] = mean(vals)
                    changed = true
                end
            end
        end

        A = B
        if !changed
            break
        end
    end
    println("done.")
    return A
end

function fill_nan_neighbors_2d!(A; maxiter=100, radius=2)
    print("Filling NaNs in 2D...")
    A = copy(A)
    nx, ny = size(A)

    for iter in 1:maxiter
        changed = false
        B = copy(A)

        for i in 1:nx, j in 1:ny
            if !isfinite(A[i,j])
                vals = Float64[]
                for di in -radius:radius, dj in -radius:radius
                    ii, jj = i+di, j+dj
                    if 1 <= ii <= nx && 1 <= jj <= ny && isfinite(A[ii,jj])
                        push!(vals, A[ii,jj])
                    end
                end
                if !isempty(vals)
                    B[i,j] = mean(vals)
                    changed = true
                end
            end
        end

        A = B
        if !changed
            break
        end
    end
    println("done.")
    return A
end

"""
    build_uniform_bspline_3d(x, y, z, B; nx, ny, nz, 
                             xuniform=nothing, yuniform=nothing, zuniform=nothing)

Take a 3D field B defined on a possibly nonuniform rectilinear grid (x,y,z),
resample it onto a uniform grid, and return:
- the uniform grids (xu, yu, zu)
- the resampled array Bu
- a cubic B-spline interpolant on the uniform grid

Assumes B has size (length(x), length(y), length(z)).
"""
function build_uniform_bspline_3d(x, y, z, B; nx=length(x), ny=length(y), nz=length(z),
                                  xuniform=nothing, yuniform=nothing, zuniform=nothing)

    print("Building uniform bspline interpolation model in 3D...")
    @assert size(B) == (length(x), length(y), length(z)) "B must have size (length(x), length(y), length(z))"
    @assert all(isfinite, B) "B still contains NaNs or Infs"

    # --- temporary interpolant on the original grid ---
    # This works for nonuniform rectilinear grids
    tmp_itp = interpolate((x, y, z), B, Gridded(Linear()))

    # --- build uniform grids ---
    xu = xuniform === nothing ? range(first(x), last(x), length=nx) : xuniform
    yu = yuniform === nothing ? range(first(y), last(y), length=ny) : yuniform
    zu = zuniform === nothing ? range(first(z), last(z), length=nz) : zuniform

    # --- sample on the uniform grid ---
    Bu = Array{Float64}(undef, length(xu), length(yu), length(zu))
    for i in eachindex(xu), j in eachindex(yu), k in eachindex(zu)
        Bu[i, j, k] = tmp_itp(xu[i], yu[j], zu[k])
    end

    # --- build cubic B-spline on the uniform grid ---
    itp = interpolate(Bu, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, xu, yu, zu)
    println("done.")

    return xu, yu, zu, Bu, sitp
end

"""
    build_uniform_bspline_2d(x, y, bs; nx, ny, xuniform=nothing, yuniform=nothing)

Take a 2D field bs defined on a possibly nonuniform rectilinear grid (x,y),
resample it onto a uniform grid, and return:
- the uniform grids (xu, yu)
- the resampled array bsu
- a cubic B-spline interpolant on the uniform grid
"""
function build_uniform_bspline_2d(x, y, bs; nx=length(x), ny=length(y),
                                  xuniform=nothing, yuniform=nothing)
    print("Building uniform bspline interpolation model in 2D")
    @assert size(bs) == (length(x), length(y)) "bs must have size (length(x), length(y))"
    @assert all(isfinite, bs) "bs still contains NaNs or Infs"

    tmp_itp = interpolate((x, y), bs, Gridded(Linear()))

    xu = xuniform === nothing ? range(first(x), last(x), length=nx) : xuniform
    yu = yuniform === nothing ? range(first(y), last(y), length=ny) : yuniform

    bsu = Array{Float64}(undef, length(xu), length(yu))
    for i in eachindex(xu), j in eachindex(yu)
        bsu[i, j] = tmp_itp(xu[i], yu[j])
    end

    itp = interpolate(bsu, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, xu, yu)
    println("done.")

    return xu, yu, bsu, sitp
end

function interp_error_stats(truevals, approxvals)
    ok = isfinite.(truevals) .& isfinite.(approxvals)
    if count(ok) == 0
        return (maxabs=NaN, rmse=NaN, mae=NaN)
    end
    err = approxvals[ok] .- truevals[ok]
    return (
        maxabs = maximum(abs.(err)),
        rmse   = sqrt(mean(err.^2)),
        mae    = mean(abs.(err))
    )
end

function check_3d_interpolation_accuracy(x, y, z, Bsp, Borig;
    nsamp=10_000, seed=1234)

    print("Running check_3d_interpolation_accuracy...")
    rng = MersenneTwister(seed)

    xmin, xmax = first(x), last(x)
    ymin, ymax = first(y), last(y)
    zmin, zmax = first(z), last(z)

    truevals = Float64[]
    approxvals = Float64[]

    for _ in 1:nsamp
        xr = xmin + rand(rng) * (xmax - xmin)
        yr = ymin + rand(rng) * (ymax - ymin)
        zr = zmin + rand(rng) * (zmax - zmin)

        tv = Borig(xr, yr, zr)
        av = Bsp(xr, yr, zr)

        if isfinite(tv) && isfinite(av)
            push!(truevals, tv)
            push!(approxvals, av)
        end
    end

    stats = interp_error_stats(truevals, approxvals)
    println("done.")

    return stats, truevals, approxvals
end

function check_vertical_profile(x0, y0, z, B, Bsp, xgrid, ygrid)
    # choose nearest original grid point
    ix = argmin(abs.(xgrid .- x0))
    iy = argmin(abs.(ygrid .- y0))

    trueprof = [B[ix, iy, k] for k in 1:length(z)]
    approxprof = [Bsp(x0, y0, zk) for zk in z]

    stats = interp_error_stats(trueprof, approxprof)
    return stats, trueprof, approxprof
end

function check_grid_reproduction_3d(xu, yu, zu, Bu, Bsp)
    nx, ny, nz = size(Bu)
    err = Array{Float64}(undef, nx, ny, nz)

    for i in 1:nx, j in 1:ny, k in 1:nz
        err[i, j, k] = Bsp(xu[i], yu[j], zu[k]) - Bu[i, j, k]
    end

    return (
        maxabs = maximum(abs.(err)),
        rmse   = sqrt(mean(err.^2)),
        mae    = mean(abs.(err))
    ), err
end

function surface_slice(Bsp, xu, yu, zlevel)
    nx, ny = length(xu), length(yu)
    A = Array{Float64}(undef, nx, ny)
    for i in 1:nx, j in 1:ny
        A[i, j] = Bsp(xu[i], yu[j], zlevel)
    end
    return A
end

function xz_slice(Bsp, xu, zu, y0)
    nx, nz = length(xu), length(zu)
    A = Array{Float64}(undef, nx, nz)
    for i in 1:nx, k in 1:nz
        A[i, k] = Bsp(xu[i], y0, zu[k])
    end
    return A
end

function yz_slice(Bsp, yu, zu, x0)
    ny, nz = length(yu), length(zu)
    A = Array{Float64}(undef, ny, nz)
    for j in 1:ny, k in 1:nz
        A[j, k] = Bsp(x0, yu[j], zu[k])
    end
    return A
end

function plot_north_atlantic_surface(Bsp, xu, yu, zsurf; 
    lonlim=(-100, 40), latlim=(0, 80))

    A = surface_slice(Bsp, xu, yu, zsurf)

    # mask out nonfinite values
    Aplot = copy(A)
    Aplot[.!isfinite.(Aplot)] .= NaN

    p = heatmap(
        xu, yu, Aplot',
        xlim=lonlim, ylim=latlim,
        xlabel="Longitude", ylabel="Latitude",
        title="Surface buoyancy: North Atlantic",
        color=:balance,
        aspect_ratio=:equal,
        colorbar_title="b",
        clims=(minimum(Aplot[isfinite.(Aplot)]), maximum(Aplot[isfinite.(Aplot)]))
    )

    return p
end

# ------------------------------
# User-set physical constants
# ------------------------------
const g     = 9.81
const ρ0    = 1026.0
const T0    = 0.0
const S0    = 35.0

x, y, z, T, S = read_woa_climatology("woa23_decav_t00_01.nc",
                                     "woa23_decav_s00_01.nc")

B, bs = compute_buoyancy_TEOS10(x, y, z, T, S)

Bfill  = fill_nan_neighbors!(B; radius=2)
bsfill = fill_nan_neighbors_2d!(bs; radius=2)

println("NaNs in Bfill: ", count(!isfinite, Bfill))
println("NaNs in bsfill: ", count(!isfinite, bsfill))

xu, yu, zu, Bu, Bsp = build_uniform_bspline_3d(x, y, z, Bfill; nx=128, ny=128, nz=64)
xu2, yu2, bsu, bsp = build_uniform_bspline_2d(x, y, bsfill; nx=128, ny=128)

statsprof, trueprof, approxprof = check_vertical_profile(-40.0, 30.0, z, Bfill, Bsp, x, y)
println("Vertical profile reproduction: ", statsprof)

stats_grid_3d, errgrid3d = check_grid_reproduction_3d(xu, yu, zu, Bu, Bsp)
println("3D grid reproduction: ", stats_grid_3d)

tmpB = interpolate((x, y, z), Bfill, Gridded(Linear()))
stats_rand_3d, _, _ = check_3d_interpolation_accuracy(x, y, z, Bsp, tmpB; nsamp=5000)
println("3D random-point error: ", stats_rand_3d)

pNA = plot_north_atlantic_surface(Bsp, xu, yu, zu[1])
display(pNA)