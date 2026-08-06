module OceanBuoyancyTools

using NCDatasets
using GibbsSeaWater
using BSplineKit
using Statistics
using Random
using Plots
using Infiltrator

export read_woa_climatology,
       #lonlat_to_cartesian_xy,
    #    teos10_buoyancy_density,
    #    ideal_elliptical_basin,
    #    ideal_basin_with_ridge,
    #    ideal_basin_with_constriction,
    #    ideal_exponential_buoyancy,
    #    ideal_surface_buoyancy,
    #    ideal_buoyancy_field,
    #    ideal_basin_mask,
       build_case_fields,
       build_all_woa_fields,
       run_case_diagnostics,
       OceanScales,
       nondimensionalize_case,
    #    redimensionalize_case,
       build_case_fields,
       build_nondimensional_case,
       plot_ocean_depth_nd,
       plot_surface_buoyancy_nd,
       plot_vertical_profile_comparison_nd,
       run_nd_case_diagnostics,
       print_spline_diagnostic_table,
       vertical_integrated_field_from_splines,
       build_complex_horizontal_gradient_slices,
    #    build_horizontal_tensor_splines,
    #    interval_contrib,
       buoyancy_column_from_spline,
       buoyancy_profile_from_spline

const g  = 9.81
const ρ0 = 1026.0

include("src/idealized.jl")
include("src/woa.jl")
include("src/interp.jl")
include("src/diagnostics.jl")
include("src/cases.jl")
include("src/NonDimensionalizationTools.jl")
include("src/TensorSplines.jl")
include("src/GreensFunction.jl")

end # module