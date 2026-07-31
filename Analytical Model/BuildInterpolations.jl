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