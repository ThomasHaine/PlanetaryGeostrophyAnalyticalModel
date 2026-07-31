include("OceanBuoyancyTools.jl")
using .OceanBuoyancyTools

fields = build_all_woa_fields(
    "woa23_decav_t00_01.nc",
    "woa23_decav_s00_01.nc";
    surface_level=:deepest,
    nx=256, ny=128, nz=64
)

println(basic_field_tests(fields))

stats, p = sample_profile_test(fields; x0=-30.0, y0=40.0)
println(stats)
display(p)

p2 = plot_sample_vertical_profiles(fields, [(-30.0, 40.0), (-40.0, 30.0)])
display(p2)