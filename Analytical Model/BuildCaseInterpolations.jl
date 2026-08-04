using Revise
include("OceanBuoyancyTools.jl")

# --------------------------------------------------------------------------
# User choice
# --------------------------------------------------------------------------
case = :ideal_elliptical
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
    fields = OceanBuoyancyTools.build_case_fields(:woa;
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

    fields = OceanBuoyancyTools.build_case_fields(:ideal_elliptical;
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

    fields = OceanBuoyancyTools.build_case_fields(:ideal_ridge;
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

    fields = OceanBuoyancyTools.build_case_fields(:ideal_constriction;
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
println(OceanBuoyancyTools.basic_field_tests(fields))

stats, p = OceanBuoyancyTools.sample_profile_test(fields)
println(stats)
display(p)

# p2 = OceanBuoyancyTools.plot_sample_vertical_profiles(fields, sample_points)
# display(p2)

# General diagnostic plots
OceanBuoyancyTools.run_case_diagnostics(fields;
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
    ν0 = 1.0e-2,
    κ0 = 1.0e-3,
    γ  = 1.0e-5,
    τ0 = 0.1
)

fields_nd = OceanBuoyancyTools.build_nondimensional_case(fields, S)
# p = OceanBuoyancyTools.plot_vertical_profile_comparison_nd(
#     0.0, 0.0, fields_nd.ẑ,
#     fields_nd.B̂fill, fields_nd.B̂zsplines,
#     fields_nd.x̂, fields_nd.ŷ
# )
# display(p)

# p2 = OceanBuoyancyTools.plot_surface_buoyancy_nd(fields_nd.b̂sfill, fields_nd.x̂, fields_nd.ŷ, fields_nd.Ĥ)
# display(p2)

# p3 = OceanBuoyancyTools.plot_ocean_depth_nd(fields_nd.Ĥ,fields_nd.x̂, fields_nd.ŷ)
# display(p3)

# results, plots = OceanBuoyancyTools.run_nd_case_diagnostics(fields_nd; sample_points=[(0.1, 0.2), (0.5, 0.5)])
# OceanBuoyancyTools.print_spline_diagnostic_table(results)


@time J = OceanBuoyancyTools.vertical_integrated_field_from_splines(fields_nd.B̂zsplines, fields_nd.ẑ)
@time Q, coeffs, basis = OceanBuoyancyTools.build_complex_horizontal_gradient_slices(fields_nd.x̂, fields_nd.ŷ, J)