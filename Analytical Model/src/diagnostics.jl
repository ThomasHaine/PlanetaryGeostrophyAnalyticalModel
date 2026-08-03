using Printf

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
    check_vertical_profile(x0, y0, z, F, xgrid, ygrid)

Compare a vertical profile from field `F` at the nearest
horizontal grid point.
"""
function check_vertical_profile(x0, y0, z, F, xgrid, ygrid)
    ix = argmin(abs.(xgrid .- x0))
    iy = argmin(abs.(ygrid .- y0))

    trueprof = [F[ix, iy, k] for k in 1:length(z)]
    return trueprof, ix, iy
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
        title = "Ocean basin depth",
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
    plot_vertical_profile_comparison(x0, y0, z, B, T, S, ρ, xgrid, ygrid; use_logz=false)

Plot vertical profile comparisons for buoyancy, temperature, salinity, and density
at a given horizontal location.
"""
function plot_vertical_profile_comparison(x0, y0, z, B, T, S, ρ, xgrid, ygrid; use_logz=false)
    ix = argmin(abs.(xgrid .- x0))
    iy = argmin(abs.(ygrid .- y0))

    trueB = [B[ix, iy, k] for k in 1:length(z)]
    trueT = [T[ix, iy, k] for k in 1:length(z)]
    trueS = [S[ix, iy, k] for k in 1:length(z)]
    trueρ = [ρ[ix, iy, k] for k in 1:length(z)]

    p = plot(layout = (1, 4), size = (1800, 600), legend = :bottomright)

    plot!(p[1], trueB, z, label = "Original", linewidth = 2, xlabel = "Buoyancy", ylabel = "z", title = "Buoyancy")
    plot!(p[2], trueT, z, label = "Original", linewidth = 2, xlabel = "T", ylabel = "z", title = "Temperature")
    plot!(p[3], trueS, z, label = "Original", linewidth = 2, xlabel = "S", ylabel = "z", title = "Salinity")
    plot!(p[4], trueρ, z, label = "Original", linewidth = 2, xlabel = "ρ", ylabel = "z", title = "Density")

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

    if haskey(fields, :Bu)
        results[:Bu_finite] = all(isfinite, fields.Bu)
    end
    if haskey(fields, :bsu)
        results[:bsu_finite] = all(isfinite, fields.bsu)
    end
    if haskey(fields, :Tu)
        results[:Tu_finite] = all(isfinite, fields.Tu)
    end
    if haskey(fields, :Su)
        results[:Su_finite] = all(isfinite, fields.Su)
    end
    if haskey(fields, :ρu)
        results[:ρu_finite] = all(isfinite, fields.ρu)
    end

    return results
end

"""
    sample_profile_test(fields; x0=-30.0, y0=40.0)

Check and plot a sample vertical profile.
Returns `(stats, plot)`.
"""
function sample_profile_test(fields; x0=-30.0, y0=40.0, use_logz=false)
    if haskey(fields, :Tfill) && haskey(fields, :Sfill) && haskey(fields, :ρfill)
        stats = (
            B = interp_error_stats([fields.Bfill[argmin(abs.(fields.x .- x0)),
                                                 argmin(abs.(fields.y .- y0)), k] for k in 1:length(fields.z)],
                                   [fields.Bfill[argmin(abs.(fields.x .- x0)),
                                                 argmin(abs.(fields.y .- y0)), k] for k in 1:length(fields.z)]),
            T = interp_error_stats([fields.Tfill[argmin(abs.(fields.x .- x0)),
                                                 argmin(abs.(fields.y .- y0)), k] for k in 1:length(fields.z)],
                                   [fields.Tfill[argmin(abs.(fields.x .- x0)),
                                                 argmin(abs.(fields.y .- y0)), k] for k in 1:length(fields.z)]),
            S = interp_error_stats([fields.Sfill[argmin(abs.(fields.x .- x0)),
                                                 argmin(abs.(fields.y .- y0)), k] for k in 1:length(fields.z)],
                                   [fields.Sfill[argmin(abs.(fields.x .- x0)),
                                                 argmin(abs.(fields.y .- y0)), k] for k in 1:length(fields.z)]),
            ρ = interp_error_stats([fields.ρfill[argmin(abs.(fields.x .- x0)),
                                                 argmin(abs.(fields.y .- y0)), k] for k in 1:length(fields.z)],
                                   [fields.ρfill[argmin(abs.(fields.x .- x0)),
                                                 argmin(abs.(fields.y .- y0)), k] for k in 1:length(fields.z)])
        )

        p = plot_vertical_profile_comparison(
            x0, y0, fields.z,
            fields.Bfill, fields.Tfill, fields.Sfill, fields.ρfill,
            fields.x, fields.y;
            use_logz = use_logz
        )
        return stats, p
    else
        stats = (
            B = interp_error_stats([fields.Bfill[argmin(abs.(fields.x .- x0)),
                                                 argmin(abs.(fields.y .- y0)), k] for k in 1:length(fields.z)],
                                   [fields.Bfill[argmin(abs.(fields.x .- x0)),
                                                 argmin(abs.(fields.y .- y0)), k] for k in 1:length(fields.z)])
        )
        p = plot_sample_vertical_profiles(fields, [(x0, y0)])
        return stats, p
    end
end

function plot_sample_vertical_profiles(fields, points; use_logz=false)
    p = plot(layout = (length(points), 1), size = (900, 300*length(points)), legend = :bottomright)

    for (i, (x0, y0)) in enumerate(points)
        ix = argmin(abs.(fields.x .- x0))
        iy = argmin(abs.(fields.y .- y0))
        trueB = [fields.Bfill[ix, iy, k] for k in 1:length(fields.z)]

        plot!(p[i], trueB, fields.z, label = "B", linewidth = 2)
        title!(p[i], "Profile at ($(x0), $(y0))")
        xlabel!(p[i], "buoyancy")
        ylabel!(p[i], "z")
        if use_logz
            yaxis!(p[i], :log)
        end
    end

    return p
end

function run_case_diagnostics(fields; lonlim=nothing, latlim=nothing, sample_points=Tuple[])
    plots_out = Any[]

    if haskey(fields, :depth2d)
        push!(plots_out, plot_ocean_depth_masked(fields.depth2d, fields.x, fields.y;
                                                 lonlim=lonlim, latlim=latlim))
        push!(plots_out, plot_north_atlantic_surface_masked(fields.Bu, fields.xu, fields.yu, fields.depth2d;
                                                            lonlim=lonlim, latlim=latlim))
    elseif haskey(fields, :H)
        push!(plots_out, plot_ideal_basin(fields.x, fields.y, fields.H;
                                          lonlim=lonlim, latlim=latlim))
        A = mask_land(fields.Bu[:, :, 1], interp2d_on_grid(fields.H, fields.x, fields.y, fields.xu, fields.yu))
        vals = A[isfinite.(A)]
        push!(plots_out, heatmap(fields.xu, fields.yu, A',
                                 title = "Surface buoyancy",
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
        end
    end

    for p in plots_out
        display(p)
    end

    return plots_out
end

"""
    eval_vertical_spline_column(spl, zgrid)

Evaluate a 1D spline object on a vertical grid.
"""
function eval_vertical_spline_column(spl, zgrid)
    return [spl(z) for z in zgrid]
end

"""
    central_diff_1d(f, z)

Compute centered finite differences of a 1D sampled profile.
Returns an array of the same length as `f`.
Endpoints use one-sided differences.
"""
function central_diff_1d(f, z)
    n = length(f)
    @assert length(z) == n
    df = similar(f)

    if n < 2
        return fill(NaN, n)
    end

    df[1] = (f[2] - f[1]) / (z[2] - z[1])
    @inbounds for k in 2:n-1
        df[k] = (f[k+1] - f[k-1]) / (z[k+1] - z[k-1])
    end
    df[n] = (f[n] - f[n-1]) / (z[n] - z[n-1])

    return df
end

"""
    trapz_1d(f, z)

Compute the trapezoidal integral of a sampled 1D profile.
"""
function trapz_1d(f, z)
    @assert length(f) == length(z)
    s = 0.0
    @inbounds for k in 1:length(z)-1
        s += 0.5 * (f[k] + f[k+1]) * (z[k+1] - z[k])
    end
    return s
end

"""
    spline_profile_stats(z, f, spl)

Compare a sampled profile `f(z)` against spline `spl(z)`.

Returns statistics for values, derivatives, and integrals.
"""
function spline_profile_stats(z, f, spl)
    approx = eval_vertical_spline_column(spl, z)
    valstats = interp_error_stats(f, approx)

    # Derivative comparison
    dfdz_fd = central_diff_1d(f, z)
    dfdz_sp = [BSplineKit.diff(spl)(zk) for zk in z]
    derivstats = interp_error_stats(dfdz_fd, dfdz_sp)

    # Integral comparison
    int_fd = trapz_1d(f, z)
    int_sp = spline_definite_integral(spl, first(z), last(z))

    return (
        values = valstats,
        derivative = derivstats,
        integral_abs_error = abs(int_sp - int_fd),
        integral_fd = int_fd,
        integral_sp = int_sp
    )
end

"""
    check_vertical_spline_field_3d(Bu, Bzsplines, zgrid)

Compare every spline column in a 3D field against the resampled grid data.

Returns summary stats for:
- values
- derivatives
- integrals
"""
function check_vertical_spline_field_3d(Bu, Bzsplines, zgrid)
    nx, ny, nz = size(Bu)

    val_true = Float64[]
    val_approx = Float64[]
    d_true = Float64[]
    d_approx = Float64[]
    int_err = Float64[]

    @inbounds for i in 1:nx, j in 1:ny
        f = vec(Bu[i, j, :])
        spl = Bzsplines[i, j]

        # values
        approx = eval_vertical_spline_column(spl, zgrid)
        append!(val_true, f)
        append!(val_approx, approx)

        # derivatives
        fd = central_diff_1d(f, zgrid)
        ad = [BSplineKit.diff(spl)(zk) for zk in zgrid]
        append!(d_true, fd)
        append!(d_approx, ad)

        # integrals
        int_fd = trapz_1d(f, zgrid)
        int_sp = spline_definite_integral(spl, first(zgrid), last(zgrid))
        push!(int_err, abs(int_sp - int_fd))
    end

    return (
        values = interp_error_stats(val_true, val_approx),
        derivatives = interp_error_stats(d_true, d_approx),
        integral_abs_error = (
            maxabs = maximum(int_err),
            rmse = sqrt(mean(int_err.^2)),
            mae = mean(abs.(int_err))
        )
    )
end

"""
    check_surface_spline_field_2d(bsu, bssplines, ygrid)

Compare every spline row against the resampled 2D field data.
"""
function check_surface_spline_field_2d(bsu, bssplines, ygrid)
    nx, ny = size(bsu)

    val_true = Float64[]
    val_approx = Float64[]
    d_true = Float64[]
    d_approx = Float64[]
    int_err = Float64[]

    @inbounds for i in 1:nx
        f = vec(bsu[i, :])
        spl = bssplines[i]

        approx = [spl(y) for y in ygrid]
        append!(val_true, f)
        append!(val_approx, approx)

        fd = central_diff_1d(f, ygrid)
        ad = [BSplineKit.diff(spl)(yk) for yk in ygrid]
        append!(d_true, fd)
        append!(d_approx, ad)

        int_fd = trapz_1d(f, ygrid)
        int_sp = spline_definite_integral(spl, first(ygrid), last(ygrid))
        push!(int_err, abs(int_sp - int_fd))
    end

    return (
        values = interp_error_stats(val_true, val_approx),
        derivatives = interp_error_stats(d_true, d_approx),
        integral_abs_error = (
            maxabs = maximum(int_err),
            rmse = sqrt(mean(int_err.^2)),
            mae = mean(abs.(int_err))
        )
    )
end

"""
    plot_vertical_spline_validation(x̂0, ŷ0, ẑ, B̂u, B̂zsplines, x̂grid, ŷgrid)

Plot a profile and its spline reconstruction in nondimensional space.
"""
function plot_vertical_spline_validation(x̂0, ŷ0, ẑ, B̂u, B̂zsplines, x̂grid, ŷgrid)
    ix = argmin(abs.(x̂grid .- x̂0))
    iy = argmin(abs.(ŷgrid .- ŷ0))

    trueB = vec(B̂u[ix, iy, :])
    spl = B̂zsplines[ix, iy]
    approxB = [spl(zk) for zk in ẑ]

    ok = isfinite.(trueB) .& isfinite.(approxB)

    p = plot(
        trueB[ok], ẑ[ok],
        label = "Grid",
        linewidth = 2,
        xlabel = "b̂",
        ylabel = "ẑ",
        title = "Vertical spline validation",
        legend = :bottomright
    )
    plot!(p, approxB[ok], ẑ[ok], label = "Spline", linewidth = 2, linestyle = :dash)
    return p
end

"""
    plot_vertical_derivative_validation(x̂0, ŷ0, ẑ, B̂u, B̂zsplines, x̂grid, ŷgrid)

Compare analytic spline derivative to finite differences.
"""
function plot_vertical_derivative_validation(x̂0, ŷ0, ẑ, B̂u, B̂zsplines, x̂grid, ŷgrid)
    ix = argmin(abs.(x̂grid .- x̂0))
    iy = argmin(abs.(ŷgrid .- ŷ0))

    f = vec(B̂u[ix, iy, :])
    spl = B̂zsplines[ix, iy]

    fd = central_diff_1d(f, ẑ)
    ad = [BSplineKit.diff(spl)(zk) for zk in ẑ]

    ok = isfinite.(fd) .& isfinite.(ad)

    p = plot(
        fd[ok], ẑ[ok],
        label = "Finite diff",
        linewidth = 2,
        xlabel = "db̂/dẑ",
        ylabel = "ẑ",
        title = "Vertical derivative validation",
        legend = :bottomright
    )
    plot!(p, ad[ok], ẑ[ok], label = "Spline derivative", linewidth = 2, linestyle = :dash)
    return p
end

"""
    run_nd_case_diagnostics(fields_nd; sample_points=Tuple[])

Run validation checks for a nondimensional case.
"""
function run_nd_case_diagnostics(fields_nd; sample_points=Tuple[])
    results = Dict{Symbol, Any}()
    plots_out = Any[]

    # 3D buoyancy spline checks
    if haskey(fields_nd, :B̂u) && haskey(fields_nd, :B̂zsplines)
        results[:B̂_spline_stats] = check_vertical_spline_field_3d(
            fields_nd.B̂u, fields_nd.B̂zsplines, fields_nd.ẑu
        )

        if !isempty(sample_points)
            for (x0, y0) in sample_points
                push!(plots_out, plot_vertical_spline_validation(
                    x0, y0, fields_nd.ẑu, fields_nd.B̂u, fields_nd.B̂zsplines, fields_nd.x̂u, fields_nd.ŷu
                ))
                push!(plots_out, plot_vertical_derivative_validation(
                    x0, y0, fields_nd.ẑu, fields_nd.B̂u, fields_nd.B̂zsplines, fields_nd.x̂u, fields_nd.ŷu
                ))
            end
        end
    end

    # 2D surface buoyancy checks
    if haskey(fields_nd, :b̂su) && haskey(fields_nd, :b̂ssplines)
        results[:b̂s_spline_stats] = check_surface_spline_field_2d(
            fields_nd.b̂su, fields_nd.b̂ssplines, fields_nd.ŷu2
        )
    end

    return results, plots_out
end

function print_spline_diagnostic_table(stats_dict)
    rows = Vector{Vector{String}}()

    for (name, stats) in sort(collect(stats_dict); by = x -> String(x[1]))
        if !(stats isa NamedTuple)
            continue
        end

        v = hasproperty(stats, :values) ? stats.values : (maxabs=NaN, rmse=NaN, mae=NaN)
        d = hasproperty(stats, :derivatives) ? stats.derivatives : (maxabs=NaN, rmse=NaN, mae=NaN)
        i = hasproperty(stats, :integral_abs_error) ? stats.integral_abs_error : (maxabs=NaN, rmse=NaN, mae=NaN)

        push!(rows, [
            String(name),
            @sprintf("%.3e", v.maxabs),
            @sprintf("%.3e", v.rmse),
            @sprintf("%.3e", v.mae),
            @sprintf("%.3e", d.maxabs),
            @sprintf("%.3e", d.rmse),
            @sprintf("%.3e", d.mae),
            @sprintf("%.3e", i.maxabs),
            @sprintf("%.3e", i.rmse),
            @sprintf("%.3e", i.mae)
        ])
    end

    if isempty(rows)
        println("No spline diagnostics found to print.")
        return nothing
    end

    headers = [
        "Field",
        "Value max|e|", "Value RMSE", "Value MAE",
        "Deriv max|e|", "Deriv RMSE", "Deriv MAE",
        "Int max|e|", "Int RMSE", "Int MAE"
    ]

    # compute column widths
    ncol = length(headers)
    widths = [length(h) for h in headers]
    for r in rows
        for j in 1:ncol
            widths[j] = max(widths[j], length(r[j]))
        end
    end

    # helper to print a separator
    function print_sep()
        print("+")
        for w in widths
            print(repeat("-", w + 2), "+")
        end
        println()
    end

    # header
    print_sep()
    print("|")
    for j in 1:ncol
        print(" ", rpad(headers[j], widths[j]), " |")
    end
    println()
    print_sep()

    # rows
    for r in rows
        print("|")
        for j in 1:ncol
            print(" ", rpad(r[j], widths[j]), " |")
        end
        println()
    end
    print_sep()

    return nothing
end