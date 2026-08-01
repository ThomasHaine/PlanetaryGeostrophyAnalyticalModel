module OceanBuoyancyTools

using NCDatasets
using GibbsSeaWater
using Interpolations
using Statistics
using Random
using Plots

export OceanScales,
       read_woa_climatology,
       lonlat_to_cartesian_xy,
       teos10_buoyancy_density,
       ideal_elliptical_basin,
       ideal_basin_with_ridge,
       ideal_basin_with_constriction,
       ideal_exponential_buoyancy,
       ideal_surface_buoyancy,
       ideal_buoyancy_field,
       ideal_basin_mask,
       build_case_fields,
       build_all_woa_fields,
       run_case_diagnostics

const g  = 9.81
const ρ0 = 1026.0

include("src/idealized.jl")
include("src/woa.jl")
include("src/interp.jl")
include("src/diagnostics.jl")
include("src/cases.jl")

end # module