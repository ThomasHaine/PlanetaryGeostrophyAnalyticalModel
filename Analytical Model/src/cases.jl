# ============================================================================
# High-level case builders
# ============================================================================

"""
    build_case_fields(case; kwargs...)

Build either WOA-based or idealized fields and their spline interpolants.

`case` may be:
- `:woa`
- `:ideal_elliptical`
- `:ideal_ridge`
- `:ideal_constriction`
"""
function build_case_fields(case::Symbol; kwargs...)
    if case == :woa
        return build_woa_case_fields(; kwargs...)
    elseif case == :ideal_elliptical
        return build_ideal_case_fields(:elliptical; kwargs...)
    elseif case == :ideal_ridge
        return build_ideal_case_fields(:ridge; kwargs...)
    elseif case == :ideal_constriction
        return build_ideal_case_fields(:constriction; kwargs...)
    else
        error("Unknown case: $case")
    end
end


# ============================================================================
# WOA case
# ============================================================================

"""
    build_woa_case_fields(; tempfile, saltfile, surface_level=:deepest,
                           nx=512, ny=256, nz=128,
                           lon0=0.0, lat0=0.0)

Build a WOA-based case in local Cartesian coordinates for interpolation/model work.

Workflow:
1. Read WOA in lon/lat/z
2. Compute buoyancy, density, and inferred bathymetry using geographic lon/lat
3. Convert lon/lat to Cartesian x/y
4. Fill NaNs
5. Build spline interpolants on uniform grids

Returns a named tuple with:
- geographic coordinates: `lon`, `lat`
- Cartesian coordinates: `x`, `y`
- vertical coordinate: `z`
- raw fields
- filled fields
- uniform grids
- interpolants
"""
function build_woa_case_fields(; tempfile,
                               saltfile,
                               surface_level=:deepest,
                               nx=512, ny=256, nz=128,
                               lon0=0.0, lat0=0.0)

    # Read WOA in geographic coordinates
    lon, lat, z, T, S = read_woa_climatology(tempfile, saltfile)

    # Compute buoyancy/density/depth using geographic lon/lat
    Bgeo, bsgeo, ρgeo, depth2dgeo = compute_buoyancy_density_depth_TEOS10(
        lon, lat, z, T, S; surface_level=surface_level
    )

    # Convert lon/lat to local Cartesian coordinates for model work
    x, y = lonlat_to_cartesian_xy(lon, lat; lon0=lon0, lat0=lat0)

    # Fill missing values before spline construction
    Bfill  = fill_nan_neighbors!(Bgeo; radius=2)
    Tfill  = fill_nan_neighbors!(T; radius=2)
    Sfill  = fill_nan_neighbors!(S; radius=2)
    ρfill  = fill_nan_neighbors!(ρgeo; radius=2)
    bsfill = fill_nan_neighbors_2d!(bsgeo; radius=2)

    # Build interpolants on uniform Cartesian grids
    xu, yu, zu, Bu, Bsp = build_uniform_bspline_3d(x, y, z, Bfill; nx=nx, ny=ny, nz=nz)
    _,  _,  _,  Tu, Tsp = build_uniform_bspline_3d(x, y, z, Tfill; nx=nx, ny=ny, nz=nz)
    _,  _,  _,  Su, Ssp = build_uniform_bspline_3d(x, y, z, Sfill; nx=nx, ny=ny, nz=nz)
    _,  _,  _,  ρu, ρsp = build_uniform_bspline_3d(x, y, z, ρfill; nx=nx, ny=ny, nz=nz)
    xu2, yu2, bsu, bsp  = build_uniform_bspline_2d(x, y, bsfill; nx=nx, ny=ny)

    return (
        case = :woa,

        # geographic coordinates
        lon = lon,
        lat = lat,

        # Cartesian model coordinates
        x = x,
        y = y,
        z = z,

        # raw fields
        T = T,
        S = S,
        B = Bgeo,
        bs = bsgeo,
        ρ = ρgeo,
        depth2d = depth2dgeo,

        # filled fields
        Bfill = Bfill,
        Tfill = Tfill,
        Sfill = Sfill,
        ρfill = ρfill,
        bsfill = bsfill,

        # uniform grids
        xu = xu,
        yu = yu,
        zu = zu,
        xu2 = xu2,
        yu2 = yu2,

        # resampled fields
        Bu = Bu,
        Tu = Tu,
        Su = Su,
        ρu = ρu,
        bsu = bsu,

        # interpolants
        Bsp = Bsp,
        Tsp = Tsp,
        Ssp = Ssp,
        ρsp = ρsp,
        bsp = bsp
    )
end


# ============================================================================
# Idealized cases
# ============================================================================

"""
    build_ideal_case_fields(kind; x, y, z, ...)

Build an idealized basin and buoyancy field, then interpolate it.

`kind` may be:
- `:elliptical`
- `:ridge`
- `:constriction`

Required keywords:
- `x`, `y`, `z`

Optional keywords:
- `nx`, `ny`, `nz`
- `buoyancy_bs`
- `buoyancy_amp`
- `buoyancy_λ`
- geometry keywords passed through to basin constructors
"""
function build_ideal_case_fields(kind::Symbol; x, y, z,
                                 nx=length(x), ny=length(y), nz=length(z),
                                 buoyancy_bs=nothing,
                                 buoyancy_amp=2e-4,
                                 buoyancy_λ=800.0,
                                 kwargs...)

    x = collect(x)
    y = collect(y)
    z = collect(z)

    # Build bathymetry
    H = if kind == :elliptical
        ideal_elliptical_basin(x, y; kwargs...)
    elseif kind == :ridge
        ideal_basin_with_ridge(x, y; kwargs...)
    elseif kind == :constriction
        ideal_basin_with_constriction(x, y; kwargs...)
    else
        error("Unknown ideal kind: $kind")
    end

    bs = buoyancy_bs === nothing ?
        ideal_surface_buoyancy(x, y; b0=-g, amp=buoyancy_amp) :
        buoyancy_bs

    B = ideal_buoyancy_field(x, y, z;
        bs = bs,
        bdeep = -9.83,
        a1 = 0.004,
        λ1 = 100.0,
        a2 = 0.006,
        λ2 = 500.0,
        a3 = 0.002,
        λ3 = 1800.0
    )

    @assert size(bs) == (length(x), length(y))

    # Mask outside ocean
    mask = ideal_basin_mask(x, y, z, H)
    Bmasked = copy(B)
    Bmasked[.!mask] .= NaN

    # Fill NaNs for spline construction
    Bfill = fill_nan_neighbors!(Bmasked; radius=2)
    bsfill = fill_nan_neighbors_2d!(bs; radius=2)

    # Build interpolants on uniform grids
    xu, yu, zu, Bu, Bsp = build_uniform_bspline_3d(x, y, z, Bfill; nx=nx, ny=ny, nz=nz)
    xu2, yu2, bsu, bsp  = build_uniform_bspline_2d(x, y, bsfill; nx=nx, ny=ny)

    return (
        case = kind,

        # original grid and fields
        x = x,
        y = y,
        z = z,
        H = H,
        bs = bs,
        B = Bmasked,

        # filled fields
        Bfill = Bfill,
        bsfill = bsfill,

        # uniform grids
        xu = xu,
        yu = yu,
        zu = zu,
        xu2 = xu2,
        yu2 = yu2,

        # resampled fields
        Bu = Bu,
        bsu = bsu,

        # interpolants
        Bsp = Bsp,
        bsp = bsp
    )
end