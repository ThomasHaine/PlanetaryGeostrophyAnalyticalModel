using Revise
using Plots
include("OceanBuoyancyTools.jl")

# --------------------------------------------------------------------------
# User choice
# --------------------------------------------------------------------------
case = :woa
# case = :ideal_elliptical
# case = :ideal_ridge
# case = :ideal_constriction
# case = :woa

# --------------------------------------------------------------------------
# Common plotting / diagnostic options
# --------------------------------------------------------------------------
lonlim = (-2.0e6, 2.0e6)
latlim = (-1.5e6, 1.5e6)
sample_points = [(-1.0e6, 0.0), (0.0, 5.0e5)]

# --------------------------------------------------------------------------
# Build the selected case
# --------------------------------------------------------------------------
if case == :woa
    buoyancy_fields = OceanBuoyancyTools.build_case_fields(:woa;
        tempfile = "woa23_decav_t00_01.nc",
        saltfile  = "woa23_decav_s00_01.nc",
        surface_level = :shallowest,
        nx = 256, ny = 128, nz = 64,
        lon0 = -30.0, lat0 = 40.0
    )

elseif case == :ideal_elliptical
    x = collect(range(-2.0e6, 2.0e6, length=64))
    y = collect(range(-1.5e6, 1.5e6, length=64))
    z = collect(range(-5000.0, 0.0, length=32))

    buoyancy_fields = OceanBuoyancyTools.build_case_fields(:ideal_elliptical;
        x = x, y = y, z = z,
        H0 = 4500.0,
        ax = 1.8e6, ay = 1.2e6,
        Hmin = 200.0,
        nx = 256, ny = 128, nz = 128,
        buoyancy_amp = 2e-4,
        buoyancy_λ = 800.0
    )

elseif case == :ideal_ridge
    x = collect(range(-2.0e6, 2.0e6, length=256))
    y = collect(range(-1.5e6, 1.5e6, length=128))
    z = collect(range(-5000.0, 0.0, length=128))

    buoyancy_fields = OceanBuoyancyTools.build_case_fields(:ideal_ridge;
        x = x, y = y, z = z,
        H0 = 4500.0,
        ax = 1.8e6, ay = 1.2e6,
        ridge_amp = 1200.0,
        ridge_width = 2.5e5,
        ridge_x = 0.0,
        Hmin = 200.0,
        nx = 256, ny = 128, nz = 128,
        buoyancy_amp = 2e-4,
        buoyancy_λ = 800.0
    )

elseif case == :ideal_constriction
    x = collect(range(-2.0e6, 2.0e6, length=256))
    y = collect(range(-1.5e6, 1.5e6, length=128))
    z = collect(range(-5000.0, 0.0, length=128))

    buoyancy_fields = OceanBuoyancyTools.build_case_fields(:ideal_constriction;
        x = x, y = y, z = z,
        H0 = 4500.0,
        ax = 1.8e6, ay = 1.2e6,
        constrict_dir = :y,
        constrict_center = 0.0,
        constrict_width = 2.5e5,
        constrict_height = 1200.0,
        Hmin = 200.0,
        nx = 256, ny = 128, nz = 128,
        buoyancy_amp = 2e-4,
        buoyancy_λ = 800.0
    )

else
    error("Unknown case: $case")
end

# --------------------------------------------------------------------------
# Diagnostics and figures
# --------------------------------------------------------------------------
println("Running diagnostics for case = $case")
println(OceanBuoyancyTools.basic_field_tests(buoyancy_fields))

stats, p = OceanBuoyancyTools.sample_profile_test(buoyancy_fields)
println(stats)
display(p)

# p2 = OceanBuoyancyTools.plot_sample_vertical_profiles(buoyancy_fields, sample_points)
# display(p2)

# General diagnostic plots
OceanBuoyancyTools.run_case_diagnostics(buoyancy_fields;
    lonlim = lonlim,
    latlim = latlim,
    sample_points = sample_points
)

# --------------------------------------------------------------------------
# Nondimensionalize the case
# --------------------------------------------------------------------------
S = OceanBuoyancyTools.OceanScales(
    L = 2.0e6,
    H0 = 5.0e3,
    f0 = 1.0e-4,
    N0 = 1.0e-3,
    ν0 = 1.0e-1,
    κ0 = 1.0e-1,
    γ  = 1.0e-5,
    τ0 = 0.1
)

buoyancy_fields_nd = OceanBuoyancyTools.build_nondimensional_case(buoyancy_fields, S)
# p = OceanBuoyancyTools.plot_vertical_profile_comparison_nd(
#     0.0, 0.0, buoyancy_fields_nd.ẑ,
#     buoyancy_fields_nd.B̂fill, buoyancy_fields_nd.B̂zsplines,
#     buoyancy_fields_nd.x̂, buoyancy_fields_nd.ŷ
# )
# display(p)

# p2 = OceanBuoyancyTools.plot_surface_buoyancy_nd(buoyancy_fields_nd.b̂sfill, buoyancy_fields_nd.x̂, buoyancy_fields_nd.ŷ, buoyancy_fields_nd.Ĥ)
# display(p2)

# p3 = OceanBuoyancyTools.plot_ocean_depth_nd(buoyancy_fields_nd.Ĥ,buoyancy_fields_nd.x̂, buoyancy_fields_nd.ŷ)
# display(p3)

# results, plots = OceanBuoyancyTools.run_nd_case_diagnostics(buoyancy_fields_nd; sample_points=[(0.1, 0.2), (0.5, 0.5)])
# OceanBuoyancyTools.print_spline_diagnostic_table(results)


@time J = OceanBuoyancyTools.vertical_integrated_field_from_splines(buoyancy_fields_nd.B̂zsplines, buoyancy_fields_nd.ẑ)
@time Q, coeffs, basis = OceanBuoyancyTools.build_complex_horizontal_gradient_slices(buoyancy_fields_nd.x̂, buoyancy_fields_nd.ŷ, J)


i = 30; j =  32
zspl = buoyancy_fields_nd.B̂zsplines[i,j]
this_bshat  = buoyancy_fields_nd.b̂sfill[i,j]
this_Hhat = buoyancy_fields_nd.Ĥ[i,j]
intervals, coeffs = OceanBuoyancyTools.spline_interval_power_coeffs(zspl)

function eval_power(c, x)
    c0, c1, c2, c3 = c
    return c0 + c1*x + c2*x^2 + c3*x^3
end

k = 1
a, b = intervals[k]
xmid = (a + b) / 2
@assert abs(zspl(xmid) - eval_power(coeffs[k], xmid)) < 1e-10

bcol = OceanBuoyancyTools.buoyancy_column_from_spline(
    zspl,
    buoyancy_fields_nd.ẑ,
    this_Hhat,
    S.ψ,
    S.Γ,
    S.κ,
    S.Ek,
    this_bshat;
    verbose=true
)

@show count(isfinite, bcol)
@show extrema(bcol[isfinite.(bcol)])

p = plot()
plot!(bcol, buoyancy_fields_nd.ẑ, marker=:circle)
display(p)


STOP
 
# z = -0.3
# b = OceanBuoyancyTools.buoyancy_profile_from_spline(zspl, z, this_Hhat, S.ψ, S.Γ, S.κ, S.Ek, this_bshat)

Πcol = OceanBuoyancyTools.pressure_integral_from_b_column(
    buoyancy_fields_nd.ẑ,
    bcol,
    this_Ĥ
)

p2 = plot()
plot!(Πcol, buoyancy_fields_nd.ẑ, marker=:circle)