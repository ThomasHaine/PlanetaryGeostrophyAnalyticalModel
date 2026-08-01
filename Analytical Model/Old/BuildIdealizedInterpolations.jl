using Revise
include("OceanBuoyancyTools.jl")

fields = OceanBuoyancyTools.build_case_fields(:ideal_ridge;
    x=x, y=y, z=z,
    H0=4500.0, ax=1.8e6, ay=1.2e6,
    ridge_amp=1200.0, ridge_width=2.5e5,
    ridge_x=0.0, Hmin=200.0,
    nx=256, ny=128, nz=128,
    buoyancy_amp=2e-4, buoyancy_λ=800.0
)

OceanBuoyancyTools.run_case_diagnostics(fields;
    lonlim=(-2.0e6, 2.0e6),
    latlim=(-1.5e6, 1.5e6)
)