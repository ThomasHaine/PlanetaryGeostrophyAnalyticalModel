using Revise
include("OceanBuoyancyTools.jl")

# --------------------------------------------------------------------------
# User choice
# --------------------------------------------------------------------------
case = :woa
# case = :ideal_elliptical
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
        surface_level = :deepest,
        nx = 512, ny = 256, nz = 128,
        lon0 = -30.0, lat0 = 40.0
    )

elseif case == :ideal_elliptical
    x = collect(range(-2.0e6, 2.0e6, length=256))
    y = collect(range(-1.5e6, 1.5e6, length=128))
    z = collect(range(-5000.0, 0.0, length=128))

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

stats, p = OceanBuoyancyTools.sample_profile_test(fields; x0=-30.0, y0=40.0)
println(stats)
display(p)

p2 = OceanBuoyancyTools.plot_sample_vertical_profiles(fields, sample_points)
display(p2)

# General diagnostic plots
OceanBuoyancyTools.run_case_diagnostics(fields;
    lonlim = lonlim,
    latlim = latlim,
    sample_points = sample_points
)