# ============================================================================
# WOA I/O and TEOS-10
# ============================================================================

"""
    read_woa_climatology(tempfile, saltfile; ...)

Read WOA T/S and return longitude, latitude, height z, temperature, salinity.
`z` is converted to height positive upward and sorted increasing.
"""
function read_woa_climatology(tempfile, saltfile;
    tempvar="t_an", saltvar="s_an",
    lonvar="lon", latvar="lat", zvar="depth")

    dT = NCDataset(tempfile)
    lon = Float64.(collect(dT[lonvar][:]))
    lat = Float64.(collect(dT[latvar][:]))
    z   = -Float64.(collect(dT[zvar][:]))
    Traw = Array(dT[tempvar][:, :, :, :])
    close(dT)

    dS = NCDataset(saltfile)
    Sraw = Array(dS[saltvar][:, :, :, :])
    close(dS)

    T = Float64.(coalesce.(dropdims(Traw; dims=4), NaN))
    S = Float64.(coalesce.(dropdims(Sraw; dims=4), NaN))

    p = sortperm(z)
    return lon, lat, z[p], T[:, :, p], S[:, :, p]
end

"""
    lonlat_to_cartesian_xy(lon, lat; lon0=0.0, lat0=0.0, R=6371000.0)

Convert lon/lat degrees to local Cartesian x/y meters.
"""
function lonlat_to_cartesian_xy(lon, lat; lon0=0.0, lat0=0.0, R=6371000.0)
    x = R * (π/180) .* (lon .- lon0) .* cosd(lat0)
    y = R * (π/180) .* (lat .- lat0)
    return x, y
end

"""
    teos10_buoyancy_density(SP, t, p, lon, lat)

Return buoyancy and density from TEOS-10.
"""
function teos10_buoyancy_density(SP, t, p, lon, lat)
    SA = gsw_sa_from_sp(SP, p, lon, lat)
    CT = gsw_ct_from_t(SA, t, p)
    ρ  = gsw_sigma0(SA, CT) + 1000.0
    b  = -g * ρ / ρ0
    return b, ρ
end

"""
    column_valid_range(T, S, i, j)

Return `(kmin, kmax)` where the T/S column is finite.
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
    compute_buoyancy_density_depth_TEOS10(x, y, z, T, S; surface_level=:shallowest)

Compute buoyancy, density, and inferred bathymetry from T/S.
"""
function compute_buoyancy_density_depth_TEOS10(x, y, z, T, S; surface_level=:shallowest)
    nx, ny, nz = size(T)
    @assert size(S) == size(T)

    B = fill(NaN, nx, ny, nz)
    ρ = fill(NaN, nx, ny, nz)

    @inbounds for i in 1:nx, j in 1:ny, k in 1:nz
        if isfinite(T[i,j,k]) && isfinite(S[i,j,k])
            p = gsw_p_from_z(z[k], y[j])
            b, rhoval = teos10_buoyancy_density(S[i,j,k], T[i,j,k], p, x[i], y[j])
            B[i,j,k] = b
            ρ[i,j,k] = rhoval
        end
    end

    bs = fill(NaN, nx, ny)
    depth2d = fill(NaN, nx, ny)

    @inbounds for i in 1:nx, j in 1:ny
        kmin, kmax = column_valid_range(T, S, i, j)
        if kmin !== nothing
            ksurf = surface_level === :shallowest ? kmax : kmin
            p = gsw_p_from_z(z[ksurf], y[j])
            bs[i,j], _ = teos10_buoyancy_density(S[i,j,ksurf], T[i,j,ksurf], p, x[i], y[j])
            depth2d[i,j] = -z[kmin]
        end
    end

    return B, bs, ρ, depth2d
end

"""
    build_all_woa_fields(tempfile, saltfile; ...)

Read WOA, convert to Cartesian coordinates, compute buoyancy/density/depth,
fill NaNs, and build spline interpolants.
"""
function build_all_woa_fields(tempfile, saltfile;
    surface_level=:shallowest,
    nx=512, ny=256, nz=128,
    lon0=0.0, lat0=0.0)

    lon, lat, z, T, S = read_woa_climatology(tempfile, saltfile)
    x, y = lonlat_to_cartesian_xy(lon, lat; lon0=lon0, lat0=lat0)

    B, bs, ρ, depth2d = compute_buoyancy_density_depth_TEOS10(x, y, z, T, S;
                                                              surface_level=surface_level)

    Bfill  = fill_nan_neighbors!(B; radius=2)
    Tfill  = fill_nan_neighbors!(T; radius=2)
    Sfill  = fill_nan_neighbors!(S; radius=2)
    ρfill  = fill_nan_neighbors!(ρ; radius=2)
    bsfill = fill_nan_neighbors_2d!(bs; radius=2)

    xu, yu, zu, Bu, Bsp = build_uniform_bspline_3d(x, y, z, Bfill; nx=nx, ny=ny, nz=nz)
    _,  _,  _,  Tu, Tsp = build_uniform_bspline_3d(x, y, z, Tfill; nx=nx, ny=ny, nz=nz)
    _,  _,  _,  Su, Ssp = build_uniform_bspline_3d(x, y, z, Sfill; nx=nx, ny=ny, nz=nz)
    _,  _,  _,  ρu, ρsp = build_uniform_bspline_3d(x, y, z, ρfill; nx=nx, ny=ny, nz=nz)
    xu2, yu2, bsu, bsp  = build_uniform_bspline_2d(x, y, bsfill; nx=nx, ny=ny)

    return (
        case = :woa,
        x = x, y = y, z = z,
        T = T, S = S,
        B = B, bs = bs, ρ = ρ, depth2d = depth2d,
        Bfill = Bfill, Tfill = Tfill, Sfill = Sfill, ρfill = ρfill, bsfill = bsfill,
        xu = xu, yu = yu, zu = zu,
        xu2 = xu2, yu2 = yu2,
        Bu = Bu, Tu = Tu, Su = Su, ρu = ρu, bsu = bsu,
        Bsp = Bsp, Tsp = Tsp, Ssp = Ssp, ρsp = ρsp, bsp = bsp
    )
end