# ============================================================================
# Nondimensionalization utilities
# ============================================================================

"""
    OceanScales(; L, H0, f0, N0, ν0, κ0, γ, τ0, ρ0=1026.0)

Container for the dimensional scales and derived nondimensional numbers.
"""
struct OceanScales
    L::Float64
    H0::Float64
    f0::Float64
    N0::Float64
    ν0::Float64
    κ0::Float64
    γ::Float64
    τ0::Float64
    ρ0::Float64
    U0::Float64
    p0::Float64
    b0::Float64
    Ek::Float64
    Bu::Float64
    Pr::Float64
    Γ::Float64
    δ::Float64
end

function OceanScales(; L, H0, f0, N0, ν0, κ0, γ, τ0, ρ0=1026.0)
    L  = Float64(L)
    H0 = Float64(H0)
    f0 = Float64(f0)
    N0 = Float64(N0)
    ν0 = Float64(ν0)
    κ0 = Float64(κ0)
    γ  = Float64(γ)
    τ0 = Float64(τ0)
    ρ0 = Float64(ρ0)

    U0 = (N0^2 * H0^2) / (f0 * L)
    p0 = ρ0 * f0 * L * U0
    b0 = N0^2 * H0
    Ek = ν0 / (f0 * H0^2)
    Bu = (N0^2 * H0^2) / (f0^2 * L^2)
    Pr = ν0 / κ0
    Γ  = Pr * γ / f0
    δ  = H0 / L

    return OceanScales(L, H0, f0, N0, ν0, κ0, γ, τ0, ρ0, U0, p0, b0, Ek, Bu, Pr, Γ, δ)
end

# ----------------------------------------------------------------------------
# Scalar conversions
# ----------------------------------------------------------------------------

nd_x(x, S::OceanScales) = x ./ S.L
nd_y(y, S::OceanScales) = y ./ S.L
nd_z(z, S::OceanScales) = z ./ S.H0

rd_x(x̂, S::OceanScales) = x̂ .* S.L
rd_y(ŷ, S::OceanScales) = ŷ .* S.L
rd_z(ẑ, S::OceanScales) = ẑ .* S.H0

"""
    nondimensionalize_coords(x, y, z, S)

Return nondimensional coordinates `(x̂, ŷ, ẑ)`.
"""
function nondimensionalize_coords(x, y, z, S::OceanScales)
    return nd_x(x, S), nd_y(y, S), nd_z(z, S)
end

"""
    redimensionalize_coords(x̂, ŷ, ẑ, S)

Return dimensional coordinates `(x, y, z)`.
"""
function redimensionalize_coords(x̂, ŷ, ẑ, S::OceanScales)
    return rd_x(x̂, S), rd_y(ŷ, S), rd_z(ẑ, S)
end

# ----------------------------------------------------------------------------
# Field scaling
# ----------------------------------------------------------------------------

nondimensionalize_field(F, scale) = F ./ scale
redimensionalize_field(F̂, scale) = F̂ .* scale

nondimensionalize_buoyancy(b, S::OceanScales) = b ./ S.b0
redimensionalize_buoyancy(b̂, S::OceanScales) = b̂ .* S.b0

nondimensionalize_bathymetry(H, S::OceanScales) = H ./ S.H0
redimensionalize_bathymetry(Ĥ, S::OceanScales) = Ĥ .* S.H0

nondimensionalize_depth(depth2d, S::OceanScales) = depth2d ./ S.H0
redimensionalize_depth(deptĥ, S::OceanScales) = deptĥ .* S.H0

# ----------------------------------------------------------------------------
# Named tuple helpers
# ----------------------------------------------------------------------------

"""
    nondimensionalize_case(fields, S)

Nondimensionalize a `fields` named tuple from either a WOA or ideal case.

Returns a new named tuple with nondimensional coordinates and fields.
"""
function nondimensionalize_case(fields, S::OceanScales)
    out = Dict{Symbol, Any}()

    # Coordinates
    if haskey(fields, :x)
        out[:x] = nondimensionalize_field(fields.x, S.L)
    end
    if haskey(fields, :y)
        out[:y] = nondimensionalize_field(fields.y, S.L)
    end
    if haskey(fields, :z)
        out[:z] = nondimensionalize_field(fields.z, S.H0)
    end
    if haskey(fields, :lon)
        out[:lon] = fields.lon
    end
    if haskey(fields, :lat)
        out[:lat] = fields.lat
    end

    # Raw fields
    if haskey(fields, :B)
        out[:B] = nondimensionalize_buoyancy(fields.B, S)
    end
    if haskey(fields, :bs)
        out[:bs] = nondimensionalize_buoyancy(fields.bs, S)
    end
    if haskey(fields, :H)
        out[:H] = nondimensionalize_bathymetry(fields.H, S)
    end
    if haskey(fields, :depth2d)
        out[:depth2d] = nondimensionalize_depth(fields.depth2d, S)
    end

    # Filled fields
    if haskey(fields, :Bfill)
        out[:Bfill] = nondimensionalize_buoyancy(fields.Bfill, S)
    end
    if haskey(fields, :bsfill)
        out[:bsfill] = nondimensionalize_buoyancy(fields.bsfill, S)
    end

    # If present, keep interpolants and grids unchanged in terms of object identity
    # but note they still evaluate in dimensional coordinates unless rebuilt.
    if haskey(fields, :Bsp)
        out[:Bsp] = fields.Bsp
    end
    if haskey(fields, :bsp)
        out[:bsp] = fields.bsp
    end
    if haskey(fields, :xu)
        out[:xu] = nondimensionalize_field(fields.xu, S.L)
    end
    if haskey(fields, :yu)
        out[:yu] = nondimensionalize_field(fields.yu, S.L)
    end
    if haskey(fields, :zu)
        out[:zu] = nondimensionalize_field(fields.zu, S.H0)
    end

    return (; out...)
end

"""
    redimensionalize_case(fieldŝ, S)

Redimensionalize a nondimensional case named tuple.
"""
function redimensionalize_case(fieldŝ, S::OceanScales)
    out = Dict{Symbol, Any}()

    if haskey(fieldŝ, :x)
        out[:x] = rd_x(fieldŝ.x, S)
    end
    if haskey(fieldŝ, :y)
        out[:y] = rd_y(fieldŝ.y, S)
    end
    if haskey(fieldŝ, :z)
        out[:z] = rd_z(fieldŝ.z, S)
    end
    if haskey(fieldŝ, :lon)
        out[:lon] = fieldŝ.lon
    end
    if haskey(fieldŝ, :lat)
        out[:lat] = fieldŝ.lat
    end

    if haskey(fieldŝ, :B)
        out[:B] = redimensionalize_buoyancy(fieldŝ.B, S)
    end
    if haskey(fieldŝ, :bs)
        out[:bs] = redimensionalize_buoyancy(fieldŝ.bs, S)
    end
    if haskey(fieldŝ, :H)
        out[:H] = redimensionalize_bathymetry(fieldŝ.H, S)
    end
    if haskey(fieldŝ, :depth2d)
        out[:depth2d] = redimensionalize_depth(fieldŝ.depth2d, S)
    end
    if haskey(fieldŝ, :Bfill)
        out[:Bfill] = redimensionalize_buoyancy(fieldŝ.Bfill, S)
    end
    if haskey(fieldŝ, :bsfill)
        out[:bsfill] = redimensionalize_buoyancy(fieldŝ.bsfill, S)
    end

    if haskey(fieldŝ, :Bsp)
        out[:Bsp] = fieldŝ.Bsp
    end
    if haskey(fieldŝ, :bsp)
        out[:bsp] = fieldŝ.bsp
    end
    if haskey(fieldŝ, :xu)
        out[:xu] = rd_x(fieldŝ.xu, S)
    end
    if haskey(fieldŝ, :yu)
        out[:yu] = rd_y(fieldŝ.yu, S)
    end
    if haskey(fieldŝ, :zu)
        out[:zu] = rd_z(fieldŝ.zu, S)
    end

    return (; out...)
end