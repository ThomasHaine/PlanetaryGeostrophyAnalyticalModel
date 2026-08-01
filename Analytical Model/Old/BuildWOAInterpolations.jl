using Revise
include("OceanBuoyancyTools.jl")
using .OceanBuoyancyTools

fields = build_case_fields(:woa;
    tempfile="woa23_decav_t00_01.nc",
    saltfile="woa23_decav_s00_01.nc",
    surface_level=:deepest,
    nx=512, ny=256, nz=128,
    lon0=-30.0, lat0=40.0
)

println(basic_field_tests(fields))
run_case_diagnostics(fields; lonlim=(-3.0e6, 3.0e6), latlim=(-2.0e6, 2.0e6))