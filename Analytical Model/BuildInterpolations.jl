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

    print("Reading WOA climatology from: ", tempfile, " and ", saltfile, "...")
    dT = NCDataset(tempfile)

    x = Float64.(collect(dT[lonvar][:]))
    y = Float64.(collect(dT[latvar][:]))
    z = -Float64.(collect(dT[zvar][:]))   # negative-down, but may be decreasing

    Traw = Array(dT[tempvar][:, :, :, :])
    close(dT)

    dS = NCDataset(saltfile)
    Sraw = Array(dS[saltvar][:, :, :, :])
    close(dS)

    T = dropdims(Traw; dims=4)
    S = dropdims(Sraw; dims=4)

    T = Float64.(coalesce.(T, NaN))
    S = Float64.(coalesce.(S, NaN))

    # sort z into increasing order for Interpolations.jl
    p = sortperm(z)
    z = z[p]
    T = T[:, :, p]
    S = S[:, :, p]

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

"""
    column_valid_range(T, S, i, j)

Return the index range of finite values in the vertical column at (i,j).
If no valid values exist, return (nothing, nothing).
"""
function column_valid_range(T, S, i, j)
    nz = size(T, 3)
    valid = falses(nz)
    for k in 1:nz
        valid[k] = isfinite(T[i,j,k]) && isfinite(S[i,j,k])
    end

    if !any(valid)
        return nothing, nothing
    end

    kmin = findfirst(valid)
    kmax = findlast(valid)
    return kmin, kmax
end

"""
    infer_ocean_depth_from_nans(z, T, S)

Infer local ocean depth from the deepest finite T/S values in each column.
Returns:
- depth2d: 2D array of inferred bottom depth
- kbot:    2D array of deepest valid index
- ktop:    2D array of shallowest valid index
"""
function infer_ocean_depth_from_nans(z, T, S)
    nx, ny, nz = size(T)
    @assert size(S) == size(T)

    depth2d = fill(NaN, nx, ny)
    kbot = fill(0, nx, ny)
    ktop = fill(0, nx, ny)

    for i in 1:nx, j in 1:ny
        kmin, kmax = column_valid_range(T, S, i, j)
        if kmin !== nothing
            ktop[i,j] = kmin
            kbot[i,j] = kmax
            depth2d[i,j] = -z[kmax]   # local bottom depth at deepest valid level
        end
    end

    return depth2d, ktop, kbot
end

"""
    infer_surface_level_from_nans(T, S)

Return the shallowest valid vertical index in each column.
"""
function infer_surface_level_from_nans(T, S)
    nx, ny, nz = size(T)
    ktop = fill(0, nx, ny)

    for i in 1:nx, j in 1:ny
        for k in 1:nz
            if isfinite(T[i,j,k]) && isfinite(S[i,j,k])
                ktop[i,j] = k
                break
            end
        end
    end

    return ktop
end

function teos10_density_buoyancy(SP, t, p, lon, lat)
    SA = gsw_sa_from_sp(SP, p, lon, lat)
    CT = gsw_ct_from_t(SA, t, p)
    ρ  = gsw_rho(SA, CT, p)
    b  = -g * ρ / ρ0
    return b, ρ
end

function compute_buoyancy_density_depth_TEOS10(x, y, z, T, S)
    print("Computing buoyancy using TEOS10 functions and inferring ocean depth...")
    @assert size(T) == size(S)
    nx, ny, nz = size(T)

    B = fill(NaN, nx, ny, nz)
    ρ = fill(NaN, nx, ny, nz)

    for i in 1:nx, j in 1:ny, k in 1:nz
        if isfinite(T[i, j, k]) && isfinite(S[i, j, k])
            lon = x[i]
            lat = y[j]
            p   = gsw_p_from_z(z[k], lat)
            B[i, j, k], ρ[i, j, k] = teos10_density_buoyancy(S[i, j, k], T[i, j, k], p, lon, lat)
        end
    end

    # infer local ocean top level from finite values
    bs = fill(NaN, nx, ny)
    depth2d = fill(NaN, nx, ny)

    for i in 1:nx, j in 1:ny
        kmin, kmax = column_valid_range(T, S, i, j)
        if kmin !== nothing
            lon = x[i]
            lat = y[j]

            # local "surface" buoyancy = shallowest valid ocean level
            p = gsw_p_from_z(z[kmax], lat)
            bs[i, j],_ = teos10_density_buoyancy(S[i, j, kmax], T[i, j, kmax], p, lon, lat)

            # local bottom depth = deepest valid level
            depth2d[i, j] = -z[kmin]
        end
    end
    println("done.")
    return B, bs, ρ, depth2d
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
    @assert issorted(x) "x must be strictly increasing"
    @assert issorted(y) "y must be strictly increasing"
    @assert issorted(z) "z must be strictly increasing"

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
    print("Building uniform bspline interpolation model in 2D...")
    @assert size(bs) == (length(x), length(y)) "bs must have size (length(x), length(y))"
    @assert all(isfinite, bs) "bs still contains NaNs or Infs"
    @assert issorted(x) "x must be strictly increasing"
    @assert issorted(y) "y must be strictly increasing"

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

function plot_ocean_depth(depth2d, x, y; lonlim=nothing, latlim=nothing, positive_down=true)
    dep = copy(depth2d)

    # convert to positive downward depth if requested
    if positive_down
        dep = -dep
    end

    vals = dep[isfinite.(dep)]
    @assert !isempty(vals) "No finite depth values to plot."

    clim = (minimum(vals), maximum(vals))

    p = heatmap(
        x, y, dep',
        xlabel = "Longitude",
        ylabel = "Latitude",
        title = "Inferred ocean depth",
        color = :viridis,
        colorbar_title = "Depth (m)",
        clims = clim
    )

    if lonlim !== nothing
        xlims!(p, lonlim)
    end
    if latlim !== nothing
        ylims!(p, latlim)
    end

    return p
end

function plot_vertical_profile_comparison(x0, y0, z, B, T, S, ρ, Bsp, Tsp, Ssp, ρsp, xgrid, ygrid;
    use_logz=false)

    # nearest original horizontal grid point
    ix = argmin(abs.(xgrid .- x0))
    iy = argmin(abs.(ygrid .- y0))
    println("ix,iy = ", ix, ", ", iy)

    # original profiles
    trueB  = [B[ix, iy, k] for k in 1:length(z)]
    trueT  = [T[ix, iy, k] for k in 1:length(z)]
    trueS  = [S[ix, iy, k] for k in 1:length(z)]
    trueρ  = [ρ[ix, iy, k] for k in 1:length(z)]

    # interpolated profiles
    approxB = [Bsp(x0, y0, zk) for zk in z]
    approxT = [Tsp(x0, y0, zk) for zk in z]
    approxS = [Ssp(x0, y0, zk) for zk in z]
    approxρ = [ρsp(x0, y0, zk) for zk in z]

    # finite masks
    okB = isfinite.(trueB) .& isfinite.(approxB)
    okT = isfinite.(trueT) .& isfinite.(approxT)
    okS = isfinite.(trueS) .& isfinite.(approxS)
    okρ = isfinite.(trueρ) .& isfinite.(approxρ)

    @assert any(okB) "No overlapping finite values in buoyancy profile."
    @assert any(okT) "No overlapping finite values in temperature profile."
    @assert any(okS) "No overlapping finite values in salinity profile."
    @assert any(okρ) "No overlapping finite values in density profile."

    p = plot(layout=(1,4), size=(1800, 600), legend=:bottomright)

    # Buoyancy
    plot!(p[1], trueB[okB], z[okB], label="Original", linewidth=2, xlabel="Buoyancy", ylabel="Depth (m)", title="Buoyancy")
    plot!(p[1], approxB[okB], z[okB], label="Spline", linewidth=2, linestyle=:dash)

    # Temperature
    plot!(p[2], trueT[okT], z[okT], label="Original", linewidth=2, xlabel="T (°C)", ylabel="Depth (m)", title="Temperature")
    plot!(p[2], approxT[okT], z[okT], label="Spline", linewidth=2, linestyle=:dash)

    # Salinity
    plot!(p[3], trueS[okS], z[okS], label="Original", linewidth=2, xlabel="S (psu)", ylabel="Depth (m)", title="Salinity")
    plot!(p[3], approxS[okS], z[okS], label="Spline", linewidth=2, linestyle=:dash)

    # Density
    plot!(p[4], trueρ[okρ], z[okρ], label="Original", linewidth=2, xlabel="ρ (kg m⁻³)", ylabel="Depth (m)", title="Density")
    plot!(p[4], approxρ[okρ], z[okρ], label="Spline", linewidth=2, linestyle=:dash)

    if use_logz
        for i in 1:4
            yaxis!(p[i], :log)
        end
    end

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

B, bs, ρ, depth2d = compute_buoyancy_density_depth_TEOS10(x, y, z, T, S)
println("NaNs in depth2d: ", count(!isfinite, depth2d))
println("Min depth: ", minimum(depth2d[isfinite.(depth2d)]))
println("Max depth: ", maximum(depth2d[isfinite.(depth2d)]))

pdepth = plot_ocean_depth(depth2d, x, y; lonlim=(-80, 20), latlim=(0, 70), positive_down=false)
display(pdepth)

Bfill  = fill_nan_neighbors!(B; radius=2)
println("NaNs in Bfill: ", count(!isfinite, Bfill))
Tfill  = fill_nan_neighbors!(T; radius=2)
println("NaNs in Tfill: ", count(!isfinite, Tfill))
Sfill  = fill_nan_neighbors!(S; radius=2)
println("NaNs in Sfill: ", count(!isfinite, Sfill))
ρfill  = fill_nan_neighbors!(ρ; radius=2)
println("NaNs in ρfill: ", count(!isfinite, ρfill))
bsfill = fill_nan_neighbors_2d!(bs; radius=2)
println("NaNs in bsfill: ", count(!isfinite, bsfill))

xu, yu, zu, Bu, Bsp = build_uniform_bspline_3d(x, y, z, Bfill; nx=512, ny=256, nz=128)
xu, yu, zu, Tu, Tsp = build_uniform_bspline_3d(x, y, z, Tfill; nx=512, ny=256, nz=128)
xu, yu, zu, Su, Ssp = build_uniform_bspline_3d(x, y, z, Sfill; nx=512, ny=256, nz=128)
xu, yu, zu, ρu, ρsp = build_uniform_bspline_3d(x, y, z, ρfill; nx=512, ny=256, nz=128)
xu2, yu2, bsu, bsp = build_uniform_bspline_2d(x, y, bsfill; nx=512, ny=256)

lat_check = 40.0
lon_check = -30.0
statsprof, trueprof, approxprof = check_vertical_profile(lat_check, lon_check, z, Bfill, Bsp, x, y)
println("Vertical profile reproduction at ",lon_check,"oE, ",lat_check,"oN: ", statsprof)

pprof = plot_vertical_profile_comparison(
    -30.0, 40.0, z,
    Bfill, Tfill, Sfill, ρfill,
    Bsp, Tsp, Ssp, ρsp,
    x, y
)
display(pprof)
STOP

stats_grid_3d, errgrid3d = check_grid_reproduction_3d(xu, yu, zu, Bu, Bsp)
println("3D grid reproduction: ", stats_grid_3d)

tmpB = interpolate((x, y, z), Bfill, Gridded(Linear()))
stats_rand_3d, _, _ = check_3d_interpolation_accuracy(x, y, z, Bsp, tmpB; nsamp=5000)
println("3D random-point error: ", stats_rand_3d)

pNA = plot_north_atlantic_surface(Bsp, xu, yu, zu[1])
# display(pNA)