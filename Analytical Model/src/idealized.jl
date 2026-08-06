# ============================================================================
# Idealized basins and buoyancy fields
# ============================================================================

"""
    ideal_elliptical_basin(x, y; H0=4000.0, ax=1.0e6, ay=1.0e6, x0=0.0, y0=0.0, Hmin=0.0)

Construct an idealized elliptical basin bathymetry H(x,y), positive downward.
"""
function ideal_elliptical_basin(x, y; H0=4000.0, ax=1.0e6, ay=1.0e6, x0=0.0, y0=0.0, Hmin=0.0)
    H = zeros(Float64, length(x), length(y))
    for i in eachindex(x), j in eachindex(y)
        ξ = (x[i] - x0) / ax
        η = (y[j] - y0) / ay
        r2 = ξ^2 + η^2
        H[i, j] = r2 < 1 ? Hmin + H0 * (1 - r2) : 0.0
    end
    return H
end

"""
    ideal_basin_with_ridge(x, y; ...)

Elliptical basin plus Gaussian ridge.
"""
function ideal_basin_with_ridge(x, y; H0=4000.0, ax=1.0e6, ay=1.0e6,
                                x0=0.0, y0=0.0,
                                ridge_amp=1000.0, ridge_width=2.0e5,
                                ridge_x=0.0, ridge_y=nothing,
                                Hmin=0.0)
    H = ideal_elliptical_basin(x, y; H0=H0, ax=ax, ay=ay, x0=x0, y0=y0, Hmin=Hmin)

    for i in eachindex(x), j in eachindex(y)
        if H[i, j] > 0
            ridge = ridge_y === nothing ?
                ridge_amp * exp(-((x[i] - ridge_x)^2) / (2ridge_width^2)) :
                ridge_amp * exp(-((y[j] - ridge_y)^2) / (2ridge_width^2))
            H[i, j] = max(0.0, H[i, j] - ridge)
        end
    end
    return H
end

"""
    ideal_basin_with_constriction(x, y; ...)

Elliptical basin plus a Gaussian constriction/sill.
"""
function ideal_basin_with_constriction(x, y; H0=4000.0, ax=1.0e6, ay=1.0e6,
                                       x0=0.0, y0=0.0,
                                       constrict_dir=:y,
                                       constrict_center=0.0,
                                       constrict_width=2.0e5,
                                       constrict_height=1500.0,
                                       Hmin=0.0)
    H = ideal_elliptical_basin(x, y; H0=H0, ax=ax, ay=ay, x0=x0, y0=y0, Hmin=Hmin)

    for i in eachindex(x), j in eachindex(y)
        if H[i, j] > 0
            bump = constrict_dir == :y ?
                constrict_height * exp(-((y[j] - constrict_center)^2) / (2constrict_width^2)) :
                constrict_height * exp(-((x[i] - constrict_center)^2) / (2constrict_width^2))
            H[i, j] = max(0.0, H[i, j] - bump)
        end
    end
    return H
end

"""
    ideal_surface_buoyancy(x, y; b0=0, amp=2e-3, x0=0.0, y0=0.0, ax=1.0e6, ay=1.0e6)

Gaussian surface buoyancy anomaly added to a background buoyancy `b0`.

Returns a 2D buoyancy field `b_s(x,y)`.

This is meant to be used as the upper boundary value for idealized buoyancy
fields in the same sign convention as the model:
- buoyancy near `-g` in units of m/s²
- small horizontal variations about that background
"""
function ideal_surface_buoyancy(x, y; b0=0, amp=2e-3, x0=0.0, y0=0.0,
                                ax=1.0e6, ay=1.0e6)
    bs = Array{Float64}(undef, length(x), length(y))

    for i in eachindex(x), j in eachindex(y)
        ξ = (x[i] - x0) / ax
        η = (y[j] - y0) / ay
        bs[i, j] = b0 + amp * exp(-(ξ^2 + η^2))
    end

    return bs
end

"""
    ideal_buoyancy_field(x, y, z; bs=nothing,
                         bdeep=-9.83,
                         a1=0.004, λ1=100.0,
                         a2=0.006, λ2=500.0,
                         a3=0.002, λ3=1800.0)

Construct an ideal buoyancy field `b(x,y,z)` with a realistic vertical profile.

The vertical structure is represented as a sum of exponentials:
    b_vert(z) = bdeep + a1*exp(z/λ1) + a2*exp(z/λ2) + a3*exp(z/λ3)

with `z` positive upward and `z=0` at the sea surface.

If `bs` is provided, it is used as the surface buoyancy field `b_s(x,y)`.
Otherwise a constant background surface buoyancy `bdeep` is used.

The field is then constructed so that the surface value matches `bs`:
    b(x,y,z) = bs(x,y) + [b_vert(z) - b_vert(0)]
"""
function ideal_buoyancy_field(x, y, z; bs=nothing,
                              bdeep=-9.83,
                              a1=0.004, λ1=100.0,
                              a2=0.006, λ2=500.0,
                              a3=0.002, λ3=1800.0)

    nx, ny, nz = length(x), length(y), length(z)
    B = Array{Float64}(undef, nx, ny, nz)

    # surface buoyancy field
    Bsurf = bs === nothing ? fill(bdeep, nx, ny) : bs
    @assert size(Bsurf) == (nx, ny)

    # vertical buoyancy profile relative to deep background
    bvert = similar(z, Float64)
    for k in eachindex(z)
        bvert[k] = bdeep +
                   a1 * exp(z[k] / λ1) +
                   a2 * exp(z[k] / λ2) +
                   a3 * exp(z[k] / λ3)
    end

    # shift so that the correction vanishes at the surface z = 0
    bsurfvert = bdeep + a1 * exp(0.0 / λ1) + a2 * exp(0.0 / λ2) + a3 * exp(0.0 / λ3)

    for i in 1:nx, j in 1:ny, k in 1:nz
        B[i, j, k] = Bsurf[i, j] + (bvert[k] - bsurfvert)
    end

    return B
end

"""
    ideal_basin_mask(x, y, z, H)

Boolean ocean mask: true where ocean exists.
"""
function ideal_basin_mask(x, y, z, H)
    nx, ny, nz = length(x), length(y), length(z)
    M = falses(nx, ny, nz)

    for i in 1:nx, j in 1:ny, k in 1:nz
        M[i, j, k] = (H[i, j] > 0) && (z[k] <= 0) && (z[k] >= -H[i, j])
    end
    return M
end