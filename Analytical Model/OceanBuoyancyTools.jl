module OceanBuoyancyTools

using NCDatasets
using GibbsSeaWater
using BSplineKit
using Statistics
using Random
using Plots
using LinearAlgebra
using Infiltrator

export build_case_fields,
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