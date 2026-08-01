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
    ideal_exponential_buoyancy(z; bs=0.0, Δb=1.0e-3, λ=500.0)

Exponential vertical buoyancy profile.
"""
ideal_exponential_buoyancy(z; bs=0.0, Δb=1.0e-3, λ=500.0) =
    bs .+ Δb .* (1 .- exp.(z ./ λ))

"""
    ideal_surface_buoyancy(x, y; ...)

Gaussian surface buoyancy anomaly.
"""
function ideal_surface_buoyancy(x, y; b0=0.0, amp=1.0e-3, x0=0.0, y0=0.0,
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
    ideal_buoyancy_field(x, y, z; ...)

3D buoyancy field built from a surface buoyancy field and exponential vertical structure.
"""
function ideal_buoyancy_field(x, y, z; bs=0.0, Δb=1.0e-3, λ=500.0, hx=nothing)
    nx, ny, nz = length(x), length(y), length(z)
    B = Array{Float64}(undef, nx, ny, nz)

    Bsurf = hx === nothing ? fill(bs, nx, ny) : hx
    @assert size(Bsurf) == (nx, ny)

    for i in 1:nx, j in 1:ny, k in 1:nz
        B[i, j, k] = Bsurf[i, j] + Δb * (1 - exp(z[k] / λ))
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