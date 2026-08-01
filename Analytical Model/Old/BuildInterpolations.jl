using Revise
include("OceanBuoyancyTools.jl")
using .OceanBuoyancyTools

fields = build_all_woa_fields(
    "woa23_decav_t00_01.nc",
    "woa23_decav_s00_01.nc";
    surface_level=:deepest,
    nx=512, ny=256, nz=128
)

println(basic_field_tests(fields))

stats, p = sample_profile_test(fields; x0=-30.0, y0=40.0)
println(stats)
display(p)

p2 = plot_sample_vertical_profiles(fields, [(-30.0, 40.0), (-40.0, 30.0)])
display(p2)

depth_on_spline = interp2d_on_grid(fields.depth2d, fields.x, fields.y, fields.xu, fields.yu)

pdepth = plot_ocean_depth_masked(fields.depth2d, fields.x, fields.y;
                                 lonlim=(-80, 20), latlim=(0, 70))
display(pdepth)

pNA = plot_north_atlantic_surface_masked(fields.Bsp, fields.xu, fields.yu, fields.zu[1], depth_on_spline;
                                         lonlim=(-80, 20), latlim=(0, 70))
display(pNA)

x = range(-2.0e6, 2.0e6, length=256)
y = range(-1.5e6, 1.5e6, length=128)
z = range(-5000.0, 0.0, length=128)

H = ideal_basin_with_ridge(collect(x), collect(y);
    H0=4500.0, ax=1.8e6, ay=1.2e6,
    ridge_amp=1200.0, ridge_width=2.5e5,
    ridge_x=0.0, Hmin=200.0)

pH = plot_ideal_basin(collect(x), collect(y), H)
display(pH)

bs = ideal_surface_buoyancy(collect(x), collect(y); b0=0.0, amp=2e-4, x0=0.0, y0=0.0, ax=8e5, ay=8e5)

B = ideal_buoyancy_field(collect(x), collect(y), collect(z); bs=0.0, Δb=2e-4, λ=800.0, hx=bs)

mask = ideal_basin_mask(collect(x), collect(y), collect(z), H)

Bmasked = copy(B)
Bmasked[.!mask] .= NaN

