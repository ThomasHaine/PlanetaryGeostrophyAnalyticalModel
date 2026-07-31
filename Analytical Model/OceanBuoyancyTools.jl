module OceanBuoyancyTools

using NCDatasets
using GibbsSeaWater
using Interpolations
using Statistics
using Random
using Plots

export read_woa_climatology,
       teos10_buoyancy_density,
       column_valid_range,
       infer_ocean_depth_from_nans,
       compute_buoyancy_density_depth_TEOS10,
       fill_nan_neighbors!,
       fill_nan_neighbors_2d!,
       build_uniform_bspline_3d,
       build_uniform_bspline_2d,
       interp2d_on_grid,
       interp_error_stats,
       check_3d_interpolation_accuracy,
       check_vertical_profile,
       check_grid_reproduction_3d,
       surface_slice,
       plot_north_atlantic_surface_masked,
       plot_ocean_depth_masked,
       plot_vertical_profile_comparison,
       build_all_woa_fields,
       basic_field_tests,
       sample_profile_test,
       plot_sample_vertical_profiles

# ============================================================================
# Constants
# ============================================================================

const g  = 9.81
const ρ0 = 1026.0

# ============================================================================
# WOA I/O
# ============================================================================

"""
    read_woa_climatology(tempfile, saltfile;
                         tempvar="t_an", saltvar="s_an",
                         lonvar="lon", latvar="lat", zvar="depth")

Read WOA climatological temperature and salinity from separate NetCDF files.

Convention:
- `z` is height, positive upward
- `z = 0` at the sea surface
- `z < 0` below the surface

Returns:
- `x` : longitude vector
- `y` : latitude vector
- `z` : height vector, sorted increasing
- `T` : temperature array of size `(nx, ny, nz)`
- `S` : salinity array of size `(nx, ny, nz)`
"""
function read_woa_climatology(tempfile, saltfile;
    tempvar="t_an", saltvar="s_an",
    lonvar="lon", latvar="lat", zvar="depth")

    println("Reading WOA climatology from: $tempfile and $saltfile ...")

    dT = NCDataset(tempfile)
    x = Float64.(collect(dT[lonvar][:]))
    y = Float64.(collect(dT[latvar][:]))

    # WOA depth is positive downward; convert to height positive upward
    z = -Float64.(collect(dT[zvar][:]))

    Traw = Array(dT[tempvar][:, :, :, :])
    close(dT)

    dS = NCDataset(saltfile)
    Sraw = Array(dS[saltvar][:, :, :, :])
    close(dS)

    # Remove singleton time dimension
    T = dropdims(Traw; dims=4)
    S = dropdims(Sraw; dims=4)

    # Replace missing values with NaN
    T = Float64.(coalesce.(T, NaN))
    S = Float64.(coalesce.(S, NaN))

    # Sort z into increasing order for Interpolations.jl
    p = sortperm(z)
    z = z[p]
    T = T[:, :, p]
    S = S[:, :, p]

    println("WOA read complete.")
    return x, y, z, T, S
end

"""
    inspect_nc_var(filename, varname)

Print simple metadata about a NetCDF variable.
"""
function inspect_nc_var(filename, varname)
    ds = NCDataset(filename)
    v = ds[varname]
    println("Variable: ", varname)
    println("Fieldnames: ", fieldnames(typeof(v)))
    println("Size: ", size(v))
    println("Eltype: ", eltype(v))
    close(ds)
end

# ============================================================================
# TEOS-10 helpers
# ============================================================================

"""
    teos10_buoyancy_density(SP, t, p, lon, lat)

Compute buoyancy and density from TEOS-10 inputs.

Returns `(b, ρ)`.
"""
function teos10_buoyancy_density(SP, t, p, lon, lat)
    SA = gsw_sa_from_sp(SP, p, lon, lat)
    CT = gsw_ct_from_t(SA, t, p)

    ρ = gsw_sigma0(SA, CT) + 1000.0
    b = -g * ρ / ρ0

    return b, ρ
end

"""
    column_valid_range(T, S, i, j)

Return `(kmin, kmax)` for finite values in the vertical column at `(i,j)`,
requiring both temperature and salinity to be finite.
"""
function column_valid_range(T, S, i, j)
    nz = size(T, 3)
    valid = falses(nz)

    @inbounds for k in 1:nz
        valid[k] = isfinite(T[i,j,k]) && isfinite(S[i,j,k])
    end

    if !any(valid)
        return nothing, nothing
    end

    return findfirst(valid), findlast(valid)
end

"""
    infer_ocean_depth_from_nans(z, T, S)

Infer bathymetry from finite T/S values in each column.

Convention:
- `z` is height, positive upward
- `depth2d` is positive downward, i.e. `H = -z_bottom`

Returns:
- `depth2d` : positive-down inferred bottom depth
- `ktop`    : shallowest valid vertical index
- `kbot`    : deepest valid vertical index
"""
function infer_ocean_depth_from_nans(z, T, S)
    nx, ny, nz = size(T)
    @assert size(S) == size(T)

    depth2d = fill(NaN, nx, ny)
    kbot    = fill(0, nx, ny)
    ktop    = fill(0, nx, ny)

    for i in 1:nx, j in 1:ny
        kmin, kmax = column_valid_range(T, S, i, j)
        if kmin !== nothing
            ktop[i,j] = kmin
            kbot[i,j] = kmax

            # deepest valid level = first finite index in increasing-z ordering
            depth2d[i,j] = -z[kmin]
        end
    end

    return depth2d, ktop, kbot
end

"""
    compute_buoyancy_density_depth_TEOS10(x, y, z, T, S; surface_level=:shallowest)

Compute:
- `B`      : buoyancy
- `bs`     : representative 2D buoyancy field
- `ρ`      : density
- `depth2d`: inferred bathymetry

Convention:
- `z` is height, positive upward
- `z = 0` at sea surface
- `z < 0` below sea surface
- `depth2d > 0` is depth

Arguments:
- `surface_level = :shallowest` uses the level closest to the surface (`kmax`)
- `surface_level = :deepest` uses the deepest valid level (`kmin`)
"""
function compute_buoyancy_density_depth_TEOS10(x, y, z, T, S; surface_level=:shallowest)
    println("Computing buoyancy, density, and inferred depth ...")

    @assert size(T) == size(S)
    nx, ny, nz = size(T)

    B = fill(NaN, nx, ny, nz)
    ρ = fill(NaN, nx, ny, nz)

    # 3D fields
    @inbounds for i in 1:nx, j in 1:ny, k in 1:nz
        if isfinite(T[i,j,k]) && isfinite(S[i,j,k])
            lon = x[i]
            lat = y[j]
            p   = gsw_p_from_z(z[k], lat)
            b, rhoval = teos10_buoyancy_density(S[i,j,k], T[i,j,k], p, lon, lat)
            B[i,j,k] = b
            ρ[i,j,k] = rhoval
        end
    end

    # 2D representative buoyancy and bathymetry
    bs = fill(NaN, nx, ny)
    depth2d = fill(NaN, nx, ny)

    @inbounds for i in 1:nx, j in 1:ny
        kmin, kmax = column_valid_range(T, S, i, j)
        if kmin !== nothing
            lon = x[i]
            lat = y[j]

            # Choose representative level
            ksurf = surface_level === :shallowest ? kmax : kmin

            p = gsw_p_from_z(z[ksurf], lat)
            bs[i,j], _ = teos10_buoyancy_density(S[i,j,ksurf], T[i,j,ksurf], p, lon, lat)

            # Bottom depth from deepest valid level
            depth2d[i,j] = -z[kmin]
        end
    end

    println("TEOS-10 computation complete.")
    return B, bs, ρ, depth2d
end

# ============================================================================
# NaN filling
# ============================================================================

"""
    fill_nan_neighbors!(A; maxiter=100, radius=2)

Iteratively fill NaNs in a 3D array using local neighborhood averages.
"""
function fill_nan_neighbors!(A; maxiter=100, radius=2)
    println("Filling NaNs in 3D ...")
    A = copy(A)
    nx, ny, nz = size(A)

    for _ in 1:maxiter
        changed = false
        B = copy(A)

        @inbounds for i in 1:nx, j in 1:ny, k in 1:nz
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
        !changed && break
    end

    println("3D NaN fill complete.")
    return A
end

"""
    fill_nan_neighbors_2d!(A; maxiter=100, radius=2)

Iteratively fill NaNs in a 2D array using local neighborhood averages.
"""
function fill_nan_neighbors_2d!(A; maxiter=100, radius=2)
    println("Filling NaNs in 2D ...")
    A = copy(A)
    nx, ny = size(A)

    for _ in 1:maxiter
        changed = false
        B = copy(A)

        @inbounds for i in 1:nx, j in 1:ny
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
        !changed && break
    end

    println("2D NaN fill complete.")
    return A
end

# ============================================================================
# Interpolation builders
# ============================================================================

"""
    build_uniform_bspline_3d(x, y, z, F; nx, ny, nz, xuniform=nothing, yuniform=nothing, zuniform=nothing)

Resample a 3D field `F` onto a uniform grid and build a cubic B-spline interpolant.

Returns:
- `xu, yu, zu`
- `Fu`
- `Fitp`
"""
function build_uniform_bspline_3d(x, y, z, F; nx=length(x), ny=length(y), nz=length(z),
                                  xuniform=nothing, yuniform=nothing, zuniform=nothing)

    println("Building uniform 3D B-spline interpolant ...")
    @assert size(F) == (length(x), length(y), length(z))
    @assert all(isfinite, F)
    @assert issorted(x) && issorted(y) && issorted(z)

    tmp_itp = interpolate((x, y, z), F, Gridded(Linear()))

    xu = xuniform === nothing ? range(first(x), last(x), length=nx) : xuniform
    yu = yuniform === nothing ? range(first(y), last(y), length=ny) : yuniform
    zu = zuniform === nothing ? range(first(z), last(z), length=nz) : zuniform

    Fu = Array{Float64}(undef, length(xu), length(yu), length(zu))
    @inbounds for i in eachindex(xu), j in eachindex(yu), k in eachindex(zu)
        Fu[i, j, k] = tmp_itp(xu[i], yu[j], zu[k])
    end

    itp = interpolate(Fu, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, xu, yu, zu)

    println("3D spline ready.")
    return xu, yu, zu, Fu, sitp
end

"""
    build_uniform_bspline_2d(x, y, F; nx, ny, xuniform=nothing, yuniform=nothing)

Resample a 2D field `F` onto a uniform grid and build a cubic B-spline interpolant.
"""
function build_uniform_bspline_2d(x, y, F; nx=length(x), ny=length(y),
                                  xuniform=nothing, yuniform=nothing)

    println("Building uniform 2D B-spline interpolant ...")
    @assert size(F) == (length(x), length(y))
    @assert all(isfinite, F)
    @assert issorted(x) && issorted(y)

    tmp_itp = interpolate((x, y), F, Gridded(Linear()))

    xu = xuniform === nothing ? range(first(x), last(x), length=nx) : xuniform
    yu = yuniform === nothing ? range(first(y), last(y), length=ny) : yuniform

    Fu = Array{Float64}(undef, length(xu), length(yu))
    @inbounds for i in eachindex(xu), j in eachindex(yu)
        Fu[i, j] = tmp_itp(xu[i], yu[j])
    end

    itp = interpolate(Fu, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, xu, yu)

    println("2D spline ready.")
    return xu, yu, Fu, sitp
end

function interp2d_on_grid(F, x, y, xnew, ynew)
    itp = interpolate((x, y), F, Gridded(Linear()))
    G = Array{Float64}(undef, length(xnew), length(ynew))
    for i in eachindex(xnew), j in eachindex(ynew)
        G[i,j] = itp(xnew[i], ynew[j])
    end
    return G
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
    interp_error_stats(truevals, approxvals)

Return max absolute error, RMSE, and MAE, ignoring non-finite values.
"""
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

"""
    check_3d_interpolation_accuracy(x, y, z, Fsp, Forig; nsamp=10000, seed=1234)

Compare an interpolant `Fsp` against a reference field/interpolant `Forig`
at random points in the domain.
"""
function check_3d_interpolation_accuracy(x, y, z, Fsp, Forig; nsamp=10_000, seed=1234)
    println("Running 3D interpolation accuracy check ...")
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

        tv = Forig(xr, yr, zr)
        av = Fsp(xr, yr, zr)

        if isfinite(tv) && isfinite(av)
            push!(truevals, tv)
            push!(approxvals, av)
        end
    end

    println("3D accuracy check complete.")
    return interp_error_stats(truevals, approxvals), truevals, approxvals
end

"""
    check_vertical_profile(x0, y0, z, F, Fsp, xgrid, ygrid)

Compare a vertical profile from the original field `F` at the nearest
horizontal grid point to the interpolated profile `Fsp(x0,y0,z)`.
"""
function check_vertical_profile(x0, y0, z, F, Fsp, xgrid, ygrid)
    ix = argmin(abs.(xgrid .- x0))
    iy = argmin(abs.(ygrid .- y0))

    trueprof   = [F[ix, iy, k] for k in 1:length(z)]
    approxprof = [Fsp(x0, y0, zk) for zk in z]

    return interp_error_stats(trueprof, approxprof), trueprof, approxprof
end

"""
    check_grid_reproduction_3d(xu, yu, zu, Fu, Fsp)

Check whether the spline reproduces the resampled uniform-grid values.
"""
function check_grid_reproduction_3d(xu, yu, zu, Fu, Fsp)
    nx, ny, nz = size(Fu)
    err = Array{Float64}(undef, nx, ny, nz)

    @inbounds for i in 1:nx, j in 1:ny, k in 1:nz
        err[i, j, k] = Fsp(xu[i], yu[j], zu[k]) - Fu[i, j, k]
    end

    stats = (
        maxabs = maximum(abs.(err)),
        rmse   = sqrt(mean(err.^2)),
        mae    = mean(abs.(err))
    )

    return stats, err
end

# ============================================================================
# Plotting
# ============================================================================

"""
    surface_slice(Fsp, xu, yu, zlevel)

Evaluate a 3D interpolant `Fsp` on a horizontal slice `z = zlevel`.
"""
function surface_slice(Fsp, xu, yu, zlevel)
    nx, ny = length(xu), length(yu)
    A = Array{Float64}(undef, nx, ny)

    @inbounds for i in 1:nx, j in 1:ny
        A[i, j] = Fsp(xu[i], yu[j], zlevel)
    end

    return A
end

"""
    mask_land(A, depth2d)

Return a copy of `A` with entries set to `NaN` wherever `depth2d == 0`
(or non-finite). Intended for plotting ocean-only fields.
"""
function mask_land(A, depth2d)
    Am = copy(A)
    land = .!(isfinite.(depth2d) .& (depth2d .> 0))
    Am[land] .= NaN
    return Am
end

"""
    plot_north_atlantic_surface_masked(Fsp, xu, yu, zsurf, depth2d; lonlim=(-100,40), latlim=(0,80))

Plot a surface slice over the North Atlantic, masking land using `depth2d`.
"""
function plot_north_atlantic_surface_masked(Fsp, xu, yu, zsurf, depth2d;
    lonlim=(-100, 40), latlim=(0, 80))

    A = surface_slice(Fsp, xu, yu, zsurf)

    # Mask land
    A = mask_land(A, depth2d)

    vals = A[isfinite.(A)]
    @assert !isempty(vals) "No finite values in plot region."

    return heatmap(
        xu, yu, A',
        xlim=lonlim, ylim=latlim,
        xlabel="Longitude", ylabel="Latitude",
        title="Surface buoyancy: North Atlantic",
        color=:balance,
        aspect_ratio=:equal,
        colorbar_title="b",
        clims=(minimum(vals), maximum(vals))
    )
end

function plot_ocean_depth_masked(depth2d, x, y; lonlim=nothing, latlim=nothing)
    dep = copy(depth2d)

    # Mask land and nonfinite points
    dep[.!isfinite.(dep) .| (dep .<= 0)] .= NaN

    vals = dep[isfinite.(dep)]
    @assert !isempty(vals) "No finite ocean depth values to plot."

    p = heatmap(
        x, y, dep',
        xlabel = "Longitude",
        ylabel = "Latitude",
        title = "Inferred ocean depth",
        color = :viridis,
        colorbar_title = "Depth (m)",
        clims = (minimum(vals), maximum(vals))
    )

    lonlim !== nothing && xlims!(p, lonlim)
    latlim !== nothing && ylims!(p, latlim)

    return p
end

"""
    plot_vertical_profile_comparison(x0, y0, z, B, T, S, ρ, Bsp, Tsp, Ssp, ρsp, xgrid, ygrid; use_logz=false)

Plot vertical profile comparisons for buoyancy, temperature, salinity, and density
at a given horizontal location.
"""
function plot_vertical_profile_comparison(x0, y0, z, B, T, S, ρ, Bsp, Tsp, Ssp, ρsp, xgrid, ygrid;
    use_logz=false)

    ix = argmin(abs.(xgrid .- x0))
    iy = argmin(abs.(ygrid .- y0))

    trueB = [B[ix, iy, k] for k in 1:length(z)]
    trueT = [T[ix, iy, k] for k in 1:length(z)]
    trueS = [S[ix, iy, k] for k in 1:length(z)]
    trueρ = [ρ[ix, iy, k] for k in 1:length(z)]

    approxB = [Bsp(x0, y0, zk) for zk in z]
    approxT = [Tsp(x0, y0, zk) for zk in z]
    approxS = [Ssp(x0, y0, zk) for zk in z]
    approxρ = [ρsp(x0, y0, zk) for zk in z]

    okB = isfinite.(trueB) .& isfinite.(approxB)
    okT = isfinite.(trueT) .& isfinite.(approxT)
    okS = isfinite.(trueS) .& isfinite.(approxS)
    okρ = isfinite.(trueρ) .& isfinite.(approxρ)

    @assert any(okB)
    @assert any(okT)
    @assert any(okS)
    @assert any(okρ)

    p = plot(layout=(1,4), size=(1800, 600), legend=:bottomright)

    plot!(p[1], trueB[okB], z[okB], label="Original", linewidth=2, xlabel="Buoyancy", ylabel="Depth (m)", title="Buoyancy")
    plot!(p[1], approxB[okB], z[okB], label="Spline", linewidth=2, linestyle=:dash)

    plot!(p[2], trueT[okT], z[okT], label="Original", linewidth=2, xlabel="T (°C)", ylabel="Depth (m)", title="Temperature")
    plot!(p[2], approxT[okT], z[okT], label="Spline", linewidth=2, linestyle=:dash)

    plot!(p[3], trueS[okS], z[okS], label="Original", linewidth=2, xlabel="S (psu)", ylabel="Depth (m)", title="Salinity")
    plot!(p[3], approxS[okS], z[okS], label="Spline", linewidth=2, linestyle=:dash)

    plot!(p[4], trueρ[okρ], z[okρ], label="Original", linewidth=2, xlabel="ρ (kg m⁻³)", ylabel="Depth (m)", title="Density")
    plot!(p[4], approxρ[okρ], z[okρ], label="Spline", linewidth=2, linestyle=:dash)

    if use_logz
        for i in 1:4
            yaxis!(p[i], :log)
        end
    end

    return p
end

# ============================================================================
# High-level driver and tests
# ============================================================================

"""
    build_all_woa_fields(tempfile, saltfile;
                         surface_level=:deepest,
                         nx=512, ny=256, nz=128)

Convenience function that:
1. reads WOA T/S,
2. computes buoyancy, density, and inferred depth,
3. fills NaNs,
4. builds spline interpolants.

Returns a named tuple with raw fields, filled fields, grids, and interpolants.
"""
function build_all_woa_fields(tempfile, saltfile;
    surface_level=:deepest,
    nx=512, ny=256, nz=128)

    x, y, z, T, S = read_woa_climatology(tempfile, saltfile)

    B, bs, ρ, depth2d = compute_buoyancy_density_depth_TEOS10(
        x, y, z, T, S; surface_level=surface_level
    )

    Bfill  = fill_nan_neighbors!(B; radius=2)
    Tfill  = fill_nan_neighbors!(T; radius=2)
    Sfill  = fill_nan_neighbors!(S; radius=2)
    ρfill  = fill_nan_neighbors!(ρ; radius=2)
    bsfill = fill_nan_neighbors_2d!(bs; radius=2)

    xu, yu, zu, Bu, Bsp = build_uniform_bspline_3d(x, y, z, Bfill; nx=nx, ny=ny, nz=nz)
    _,  _,  _,  Tu, Tsp = build_uniform_bspline_3d(x, y, z, Tfill; nx=nx, ny=ny, nz=nz)
    _,  _,  _,  Su, Ssp = build_uniform_bspline_3d(x, y, z, Sfill; nx=nx, ny=ny, nz=nz)
    _,  _,  _,  ρu, ρsp = build_uniform_bspline_3d(x, y, z, ρfill; nx=nx, ny=ny, nz=nz)
    xu2, yu2, bsu, bsp  = build_uniform_bspline_2d(x, y, bsfill; nx=nx, ny=ny)

    return (
        x = x, y = y, z = z,
        T = T, S = S,
        B = B, bs = bs, ρ = ρ, depth2d = depth2d,
        Bfill = Bfill, Tfill = Tfill, Sfill = Sfill, ρfill = ρfill, bsfill = bsfill,
        xu = xu, yu = yu, zu = zu,
        xu2 = xu2, yu2 = yu2,
        Bu = Bu, Tu = Tu, Su = Su, ρu = ρu, bsu = bsu,
        Bsp = Bsp, Tsp = Tsp, Ssp = Ssp, ρsp = ρsp, bsp = bsp
    )
end

"""
    basic_field_tests(fields)

Run a small set of sanity checks on the output of `build_all_woa_fields(...)`.
"""
function basic_field_tests(fields)
    results = Dict{Symbol, Any}()

    results[:Bfill_finite]  = all(isfinite, fields.Bfill)
    results[:Tfill_finite]  = all(isfinite, fields.Tfill)
    results[:Sfill_finite]  = all(isfinite, fields.Sfill)
    results[:ρfill_finite]  = all(isfinite, fields.ρfill)
    results[:bsfill_finite] = all(isfinite, fields.bsfill)

    results[:Bgrid_stats] = check_grid_reproduction_3d(fields.xu, fields.yu, fields.zu, fields.Bu, fields.Bsp)[1]
    results[:Tgrid_stats] = check_grid_reproduction_3d(fields.xu, fields.yu, fields.zu, fields.Tu, fields.Tsp)[1]
    results[:Sgrid_stats] = check_grid_reproduction_3d(fields.xu, fields.yu, fields.zu, fields.Su, fields.Ssp)[1]
    results[:ρgrid_stats] = check_grid_reproduction_3d(fields.xu, fields.yu, fields.zu, fields.ρu, fields.ρsp)[1]

    return results
end

"""
    sample_profile_test(fields; x0=-30.0, y0=40.0, use_logz=false)

Check and plot a sample vertical profile at `(x0, y0)`.

Returns `(stats, plot)`.
"""
function sample_profile_test(fields; x0=-30.0, y0=40.0, use_logz=false)
    stats = (
        B = check_vertical_profile(x0, y0, fields.z, fields.Bfill, fields.Bsp, fields.x, fields.y)[1],
        T = check_vertical_profile(x0, y0, fields.z, fields.Tfill, fields.Tsp, fields.x, fields.y)[1],
        S = check_vertical_profile(x0, y0, fields.z, fields.Sfill, fields.Ssp, fields.x, fields.y)[1],
        ρ = check_vertical_profile(x0, y0, fields.z, fields.ρfill, fields.ρsp, fields.x, fields.y)[1]
    )

    p = plot_vertical_profile_comparison(
        x0, y0, fields.z,
        fields.Bfill, fields.Tfill, fields.Sfill, fields.ρfill,
        fields.Bsp, fields.Tsp, fields.Ssp, fields.ρsp,
        fields.x, fields.y;
        use_logz=use_logz
    )

    return stats, p
end

"""
    plot_sample_vertical_profiles(fields, points; use_logz=false)

Plot sample vertical profiles at multiple `(x0, y0)` points.

`points` should be a vector of tuples, e.g.:
`[(-30.0, 40.0), (-40.0, 30.0)]`
"""
function plot_sample_vertical_profiles(fields, points; use_logz=false)
    n = length(points)
    p = plot(layout=(n, 1), size=(900, 300*n), legend=:bottomright)

    for (i, (x0, y0)) in enumerate(points)
        ix = argmin(abs.(fields.x .- x0))
        iy = argmin(abs.(fields.y .- y0))

        trueB = [fields.Bfill[ix, iy, k] for k in 1:length(fields.z)]
        approxB = [fields.Bsp(x0, y0, zk) for zk in fields.z]

        ok = isfinite.(trueB) .& isfinite.(approxB)

        plot!(p[i], trueB[ok], fields.z[ok], label="B orig", linewidth=2)
        plot!(p[i], approxB[ok], fields.z[ok], label="B spline", linewidth=2, linestyle=:dash)
        title!(p[i], "Profile at ($(x0), $(y0))")
        xlabel!(p[i], "buoyancy")
        ylabel!(p[i], "z")
        if use_logz
            yaxis!(p[i], :log)
        end
    end

    return p
end

end # module