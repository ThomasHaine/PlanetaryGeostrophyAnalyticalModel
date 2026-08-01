# ============================================================================
# Diagnostics and plotting
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
    return (maxabs = maximum(abs.(err)),
            rmse   = sqrt(mean(err.^2)),
            mae    = mean(abs.(err)))
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

    stats = (maxabs = maximum(abs.(err)),
             rmse   = sqrt(mean(err.^2)),
             mae    = mean(abs.(err)))
    return stats, err
end

# ============================================================================
# Plotting helpers
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

Mask land or non-ocean points with NaN.
Assumes positive depth `depth2d > 0` on ocean and 0/NaN on land.
"""
function mask_land(A, depth2d)
    Am = copy(A)
    land = .!(isfinite.(depth2d) .& (depth2d .> 0))
    Am[land] .= NaN
    return Am
end

"""
    plot_ideal_basin(x, y, H; lonlim=nothing, latlim=nothing)

Plot an ideal bathymetry field.
"""
function plot_ideal_basin(x, y, H; lonlim=nothing, latlim=nothing)
    vals = H[isfinite.(H) .& (H .> 0)]
    @assert !isempty(vals) "No finite positive bathymetry values."

    p = heatmap(
        x, y, H',
        xlabel = "x",
        ylabel = "y",
        title = "Ideal ocean basin depth",
        color = :viridis,
        colorbar_title = "H",
        clims = (minimum(vals), maximum(vals))
    )
    lonlim !== nothing && xlims!(p, lonlim)
    latlim !== nothing && ylims!(p, latlim)
    return p
end

"""
    plot_ocean_depth_masked(depth2d, x, y; lonlim=nothing, latlim=nothing)

Plot inferred ocean depth, masking land.
"""
function plot_ocean_depth_masked(depth2d, x, y; lonlim=nothing, latlim=nothing)
    dep = copy(depth2d)
    dep[.!isfinite.(dep) .| (dep .<= 0)] .= NaN

    vals = dep[isfinite.(dep)]
    @assert !isempty(vals) "No finite ocean depth values to plot."

    p = heatmap(
        x, y, dep',
        xlabel = "x",
        ylabel = "y",
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
    plot_north_atlantic_surface_masked(Fsp, xu, yu, zsurf, depth2d; lonlim=nothing, latlim=nothing)

Plot a surface slice, masking land using `depth2d`.
Axes are the model Cartesian coordinates x/y.
"""
function plot_north_atlantic_surface_masked(Fsp, xu, yu, zsurf, depth2d;
    lonlim=nothing, latlim=nothing)

    A = surface_slice(Fsp, xu, yu, zsurf)
    A = mask_land(A, depth2d)

    vals = A[isfinite.(A)]
    @assert !isempty(vals) "No finite values in plot region."

    p = heatmap(
        xu, yu, A',
        xlim = lonlim, ylim = latlim,
        xlabel = "x",
        ylabel = "y",
        title = "Surface buoyancy",
        color = :balance,
        aspect_ratio = :equal,
        colorbar_title = "b",
        clims = (minimum(vals), maximum(vals))
    )
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

    p = plot(layout = (1, 4), size = (1800, 600), legend = :bottomright)

    plot!(p[1], trueB[okB], z[okB], label = "Original", linewidth = 2, xlabel = "Buoyancy", ylabel = "z", title = "Buoyancy")
    plot!(p[1], approxB[okB], z[okB], label = "Spline", linewidth = 2, linestyle = :dash)

    plot!(p[2], trueT[okT], z[okT], label = "Original", linewidth = 2, xlabel = "T", ylabel = "z", title = "Temperature")
    plot!(p[2], approxT[okT], z[okT], label = "Spline", linewidth = 2, linestyle = :dash)

    plot!(p[3], trueS[okS], z[okS], label = "Original", linewidth = 2, xlabel = "S", ylabel = "z", title = "Salinity")
    plot!(p[3], approxS[okS], z[okS], label = "Spline", linewidth = 2, linestyle = :dash)

    plot!(p[4], trueρ[okρ], z[okρ], label = "Original", linewidth = 2, xlabel = "ρ", ylabel = "z", title = "Density")
    plot!(p[4], approxρ[okρ], z[okρ], label = "Spline", linewidth = 2, linestyle = :dash)

    if use_logz
        for i in 1:4
            yaxis!(p[i], :log)
        end
    end

    return p
end

# ============================================================================
# High-level diagnostics
# ============================================================================

"""
    basic_field_tests(fields)

Run a small set of sanity checks on either WOA or idealized fields.
"""
function basic_field_tests(fields)
    results = Dict{Symbol, Any}()

    if haskey(fields, :Bfill)
        results[:Bfill_finite] = all(isfinite, fields.Bfill)
    end
    if haskey(fields, :Tfill)
        results[:Tfill_finite] = all(isfinite, fields.Tfill)
    end
    if haskey(fields, :Sfill)
        results[:Sfill_finite] = all(isfinite, fields.Sfill)
    end
    if haskey(fields, :ρfill)
        results[:ρfill_finite] = all(isfinite, fields.ρfill)
    end
    if haskey(fields, :bsfill)
        results[:bsfill_finite] = all(isfinite, fields.bsfill)
    end

    if haskey(fields, :Bu) && haskey(fields, :Bsp)
        results[:Bgrid_stats] = check_grid_reproduction_3d(fields.xu, fields.yu, fields.zu, fields.Bu, fields.Bsp)[1]
    end
    if haskey(fields, :Tu) && haskey(fields, :Tsp)
        results[:Tgrid_stats] = check_grid_reproduction_3d(fields.xu, fields.yu, fields.zu, fields.Tu, fields.Tsp)[1]
    end
    if haskey(fields, :Su) && haskey(fields, :Ssp)
        results[:Sgrid_stats] = check_grid_reproduction_3d(fields.xu, fields.yu, fields.zu, fields.Su, fields.Ssp)[1]
    end
    if haskey(fields, :ρu) && haskey(fields, :ρsp)
        results[:ρgrid_stats] = check_grid_reproduction_3d(fields.xu, fields.yu, fields.zu, fields.ρu, fields.ρsp)[1]
    end

    return results
end

"""
    sample_profile_test(fields; x0=-30.0, y0=40.0)

Check and plot a sample vertical profile.

Works for both WOA and idealized fields.
Returns `(stats, plot)`.
"""
function sample_profile_test(fields; x0=-30.0, y0=40.0, use_logz=false)
    if haskey(fields, :Tfill) && haskey(fields, :Sfill) && haskey(fields, :ρfill)
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
            use_logz = use_logz
        )
        return stats, p
    else
        stats = (
            B = check_vertical_profile(x0, y0, fields.z, fields.Bfill, fields.Bsp, fields.x, fields.y)[1]
        )
        p = plot_sample_vertical_profiles(fields, [(x0, y0)])
        return stats, p
    end
end

"""
    plot_sample_vertical_profiles(fields, points)

Plot sample vertical profiles at multiple `(x0, y0)` points.
"""
function plot_sample_vertical_profiles(fields, points; use_logz=false)
    p = plot(layout = (length(points), 1), size = (900, 300*length(points)), legend = :bottomright)

    for (i, (x0, y0)) in enumerate(points)
        ix = argmin(abs.(fields.x .- x0))
        iy = argmin(abs.(fields.y .- y0))
        trueB = [fields.Bfill[ix, iy, k] for k in 1:length(fields.z)]
        approxB = [fields.Bsp(x0, y0, zk) for zk in fields.z]
        ok = isfinite.(trueB) .& isfinite.(approxB)

        plot!(p[i], trueB[ok], fields.z[ok], label = "B orig", linewidth = 2)
        plot!(p[i], approxB[ok], fields.z[ok], label = "B spline", linewidth = 2, linestyle = :dash)
        title!(p[i], "Profile at ($(x0), $(y0))")
        xlabel!(p[i], "buoyancy")
        ylabel!(p[i], "z")
        if use_logz
            yaxis!(p[i], :log)
        end
    end

    return p
end

"""
    run_case_diagnostics(fields; lonlim=nothing, latlim=nothing, sample_points=Tuple[])

Run plots and checks for either WOA or ideal cases.
Returns a vector of plots.
"""
function run_case_diagnostics(fields; lonlim=nothing, latlim=nothing, sample_points=Tuple[])
    plots_out = Any[]

    if haskey(fields, :depth2d)
        push!(plots_out, plot_ocean_depth_masked(fields.depth2d, fields.x, fields.y;
                                                 lonlim=lonlim, latlim=latlim))
        depth_on_spline = interp2d_on_grid(fields.depth2d, fields.x, fields.y, fields.xu, fields.yu)
        push!(plots_out, plot_north_atlantic_surface_masked(fields.Bsp, fields.xu, fields.yu, fields.zu[1], depth_on_spline;
                                                            lonlim=lonlim, latlim=latlim))
    elseif haskey(fields, :H)
        push!(plots_out, plot_ideal_basin(fields.x, fields.y, fields.H;
                                          lonlim=lonlim, latlim=latlim))
        H_on_spline = interp2d_on_grid(fields.H, fields.x, fields.y, fields.xu, fields.yu)
        A = mask_land(surface_slice(fields.Bsp, fields.xu, fields.yu, fields.zu[1]), H_on_spline)
        vals = A[isfinite.(A)]
        push!(plots_out, heatmap(fields.xu, fields.yu, A',
                                 title = "Surface buoyancy (ideal case)",
                                 xlabel = "x", ylabel = "y",
                                 color = :balance, aspect_ratio = :equal,
                                 clims = (minimum(vals), maximum(vals))))
    end

    if !isempty(sample_points)
        push!(plots_out, plot_sample_vertical_profiles(fields, sample_points))
    else
        if haskey(fields, :Tfill)
            _, p = sample_profile_test(fields; x0 = -30.0, y0 = 40.0)
            push!(plots_out, p)
        else
            _, p = sample_profile_test(fields; x0 = fields.x[end÷2], y0 = fields.y[end÷2])
            push!(plots_out, p)
        end
    end

    for p in plots_out
        display(p)
    end

    return plots_out
end