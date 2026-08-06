# ============================================================================
# WOA I/O and TEOS-10
# ============================================================================

"""
    read_woa_climatology(tempfile, saltfile;
                         tempvar="t_an", saltvar="s_an",
                         lonvar="lon", latvar="lat", zvar="depth")

Read WOA climatological temperature and salinity from separate NetCDF files.

Returns:
- `lon` : longitude vector in degrees
- `lat` : latitude vector in degrees
- `z`   : height vector, positive upward, sorted increasing
- `T`   : temperature array of size `(nx, ny, nz)`
- `S`   : salinity array of size `(nx, ny, nz)`

Notes:
- WOA depth is positive downward; it is converted to height positive upward.
- Missing values are converted to `NaN`.
"""
function read_woa_climatology(tempfile, saltfile;
    tempvar="t_an", saltvar="s_an",
    lonvar="lon", latvar="lat", zvar="depth")

    println("Reading WOA climatology from: $tempfile and $saltfile ...")

    dT = NCDataset(tempfile)
    lon = Float64.(collect(dT[lonvar][:]))
    lat = Float64.(collect(dT[latvar][:]))

    # WOA depth is positive downward; convert to height positive upward
    z = -Float64.(collect(dT[zvar][:]))

    Traw = Array(dT[tempvar][:, :, :, :])
    close(dT)

    dS = NCDataset(saltfile)
    Sraw = Array(dS[saltvar][:, :, :, :])
    close(dS)

    # Remove singleton time dimension and convert missings to NaN
    T = Float64.(coalesce.(dropdims(Traw; dims=4), NaN))
    S = Float64.(coalesce.(dropdims(Sraw; dims=4), NaN))

    # Sort vertical coordinate into increasing order
    p = sortperm(z)
    z = z[p]
    T = T[:, :, p]
    S = S[:, :, p]

    println("WOA read complete.")
    return lon, lat, z, T, S
end

"""
    lonlat_to_cartesian_xy(lon, lat; lon0=0.0, lat0=0.0, R=6371000.0)

Convert longitude/latitude in degrees to local Cartesian x/y in meters
using an equirectangular approximation about `(lon0, lat0)`.

Returns `(x, y)` arrays.
"""
function lonlat_to_cartesian_xy(lon, lat; lon0=0.0, lat0=0.0, R=6371000.0)
    x = R * (π / 180) .* (lon .- lon0) .* cosd(lat0)
    y = R * (π / 180) .* (lat .- lat0)
    return x, y
end

# ============================================================================
# TEOS-10 helpers
# ============================================================================

"""
    teos10_buoyancy_density(SP, t, p, lon, lat)

Compute buoyancy and density from TEOS-10 inputs.

Returns `(b, ρ)`.

Important:
- `lon`, `lat` must be geographic coordinates in degrees
- `p` is pressure in dbar
"""
function teos10_buoyancy_density(SP, t, p, lon, lat)
    SA = gsw_sa_from_sp(SP, p, lon, lat)
    CT = gsw_ct_from_t(SA, t, p)

    ρ = gsw_sigma0(SA, CT) + 1000.0
    b = -g * ρ / ρ0

    return b, ρ
end

"""
    column_valid_range(T, S, i, j)

Return `(kmin, kmax)` for finite values in the vertical column at `(i,j)`,
requiring both temperature and salinity to be finite.

Assumes `z` increases upward.
"""
function column_valid_range(T, S, i, j)
    nz = size(T, 3)
    valid = falses(nz)

    @inbounds for k in 1:nz
        valid[k] = isfinite(T[i,j,k]) && isfinite(S[i,j,k])
    end

    if !any(valid)
        return nothing, nothing
    end

    return findfirst(valid), findlast(valid)
end

"""
    compute_buoyancy_density_depth_TEOS10(lon, lat, z, T, S; surface_level=:shallowest)

Compute:
- `B`      : buoyancy
- `bs`     : representative 2D buoyancy field
- `ρ`      : density
- `depth2d`: inferred bathymetry

Important:
- `lon`, `lat` are geographic degrees
- `z` is height, positive upward
- TEOS-10 is evaluated using lon/lat, not Cartesian coordinates

Arguments:
- `surface_level = :shallowest` uses the level closest to the surface (`kmax`)
- `surface_level = :deepest` uses the deepest valid level (`kmin`)
"""
function compute_buoyancy_density_depth_TEOS10(lon, lat, z, T, S; surface_level=:shallowest)
    println("Computing buoyancy, density, and inferred depth ...")

    @assert size(T) == size(S)
    nx, ny, nz = size(T)

    B = fill(NaN, nx, ny, nz)
    ρ = fill(NaN, nx, ny, nz)

    # 3D fields
    @inbounds for i in 1:nx, j in 1:ny, k in 1:nz
        if isfinite(T[i,j,k]) && isfinite(S[i,j,k])
            p = gsw_p_from_z(z[k], lat[j])
            babs, rhoval = teos10_buoyancy_density(S[i,j,k], T[i,j,k], p, lon[i], lat[j])
            b = babs + g        # Buoyancy relative to reference density ρ0
            B[i,j,k] = b
            ρ[i,j,k] = rhoval
        end
    end

    # 2D representative buoyancy and bathymetry
    bs = fill(NaN, nx, ny)
    depth2d = fill(NaN, nx, ny)

    @inbounds for i in 1:nx, j in 1:ny
        kmin, kmax = column_valid_range(T, S, i, j)
        if kmin !== nothing
            ksurf = surface_level === :shallowest ? kmax : kmin

            p = gsw_p_from_z(z[ksurf], lat[j])
            bsabs, _ = teos10_buoyancy_density(S[i,j,ksurf], T[i,j,ksurf], p, lon[i], lat[j])
            bs[i,j] = bsabs + g        # Buoyancy relative to reference density ρ0

            # Bottom depth from deepest valid level
            depth2d[i,j] = -z[kmin]
        end
    end

    println("TEOS-10 computation complete.")
    return B, bs, ρ, depth2d
end