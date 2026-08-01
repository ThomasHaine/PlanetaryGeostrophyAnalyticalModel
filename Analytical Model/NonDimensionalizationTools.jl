module NonDimensionalizationTools

export OceanScales,
       nondimensionalize_field, redimensionalize_field,
       nondimensionalize_coords, redimensionalize_coords,
       nondimensionalize_state, redimensionalize_state,
       nondimensionalize_bathymetry, redimensionalize_bathymetry,
       nondimensionalize_buoyancy, redimensionalize_buoyancy,
       nd_x, nd_y, nd_z, nd_t,
       nd_u, nd_v, nd_w, nd_p, nd_b,
       nd_f, nd_ν, nd_κ, nd_τx, nd_τy,
       rd_x, rd_y, rd_z, rd_t,
       rd_u, rd_v, rd_w, rd_p, rd_b,
       rd_f, rd_ν, rd_κ, rd_τx, rd_τy

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

const ρ0_default = 1026.0

# -----------------------------------------------------------------------------
# Scale container
# -----------------------------------------------------------------------------

"""
    OceanScales(; L, H0, f0, N0, ν0, κ0, γ, τ0, ρ0=1026.0)

Container for dimensional scales and derived nondimensional numbers.

Parameters
----------
- `L`   : horizontal length scale
- `H0`  : depth scale
- `f0`  : Coriolis parameter scale
- `N0`  : buoyancy frequency scale
- `ν0`  : viscosity scale
- `κ0`  : diffusivity scale
- `γ`   : relaxation parameter
- `τ0`  : stress scale
- `ρ0`  : reference density

Derived
-------
- `U0`  : velocity scale, `U0 = N0^2 H0^2 / (f0 L)`
- `p0`  : pressure scale, `p0 = ρ0 f0 L U0`
- `b0`  : buoyancy scale, `b0 = N0^2 H0`
- `Ek`  : Ekman number, `ν0 / (f0 H0^2)`
- `Bu`  : Burger number, `N0^2 H0^2 / (f0^2 L^2)`
- `Pr`  : Prandtl number, `ν0 / κ0`
- `Γ`   : relaxation number, `Pr * γ / f0`
- `δ`   : aspect ratio, `H0 / L`
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

function OceanScales(; L, H0, f0, N0, ν0, κ0, γ, τ0, ρ0=ρ0_default)
    L  = Float64(L);  H0 = Float64(H0); f0 = Float64(f0); N0 = Float64(N0)
    ν0 = Float64(ν0); κ0 = Float64(κ0); γ = Float64(γ); τ0 = Float64(τ0)
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

# -----------------------------------------------------------------------------
# Scalar conversions
# -----------------------------------------------------------------------------

nd_x(x, S::OceanScales) = x / S.L
nd_y(y, S::OceanScales) = y / S.L
nd_z(z, S::OceanScales) = z / S.H0
nd_t(t, S::OceanScales) = t * S.U0 / S.L

nd_u(u, S::OceanScales) = u / S.U0
nd_v(v, S::OceanScales) = v / S.U0
nd_w(w, S::OceanScales) = w / ((S.H0 / S.L) * S.U0)

nd_p(p, S::OceanScales) = p / S.p0
nd_b(b, S::OceanScales) = b / S.b0

nd_f(f, S::OceanScales) = f / S.f0
nd_ν(ν, S::OceanScales) = ν / S.ν0
nd_κ(κ, S::OceanScales) = κ / S.κ0

nd_τx(τx, S::OceanScales) = τx / S.τ0
nd_τy(τy, S::OceanScales) = τy / S.τ0

rd_x(x̂, S::OceanScales) = x̂ * S.L
rd_y(ŷ, S::OceanScales) = ŷ * S.L
rd_z(ẑ, S::OceanScales) = ẑ * S.H0
rd_t(t̂, S::OceanScales) = t̂ * S.L / S.U0

rd_u(û, S::OceanScales) = û * S.U0
rd_v(v̂, S::OceanScales) = v̂ * S.U0
rd_w(ŵ, S::OceanScales) = ŵ * ((S.H0 / S.L) * S.U0)

rd_p(p̂, S::OceanScales) = p̂ * S.p0
rd_b(b̂, S::OceanScales) = b̂ * S.b0

rd_f(f̂, S::OceanScales) = f̂ * S.f0
rd_ν(ν̂, S::OceanScales) = ν̂ * S.ν0
rd_κ(κ̂, S::OceanScales) = κ̂ * S.κ0

rd_τx(τ̂x, S::OceanScales) = τ̂x * S.τ0
rd_τy(τ̂y, S::OceanScales) = τ̂y * S.τ0

# -----------------------------------------------------------------------------
# Array helpers
# -----------------------------------------------------------------------------

"""
    nondimensionalize_field(F, scale)

Return `F ./ scale`.
"""
nondimensionalize_field(F, scale) = F ./ scale

"""
    redimensionalize_field(F̂, scale)

Return `F̂ .* scale`.
"""
redimensionalize_field(F̂, scale) = F̂ .* scale

"""
    nondimensionalize_coords(x, y, z, S)

Return nondimensional coordinates `(x̂, ŷ, ẑ)`.
"""
function nondimensionalize_coords(x, y, z, S::OceanScales)
    return x ./ S.L, y ./ S.L, z ./ S.H0
end

"""
    redimensionalize_coords(x̂, ŷ, ẑ, S)

Return dimensional coordinates `(x, y, z)`.
"""
function redimensionalize_coords(x̂, ŷ, ẑ, S::OceanScales)
    return x̂ .* S.L, ŷ .* S.L, ẑ .* S.H0
end

# -----------------------------------------------------------------------------
# Full state conversions
# -----------------------------------------------------------------------------

"""
    nondimensionalize_state(u, v, w, p, b, τx, τy, S)

Convert dimensional state fields to nondimensional form.

Returns `(û, v̂, ŵ, p̂, b̂, τ̂x, τ̂y)`.
"""
function nondimensionalize_state(u, v, w, p, b, τx, τy, S::OceanScales)
    û   = u ./ S.U0
    v̂   = v ./ S.U0
    ŵ    = w ./ ((S.H0 / S.L) * S.U0)
    p̂   = p ./ S.p0
    b̂   = b ./ S.b0
    τ̂x  = τx ./ S.τ0
    τ̂y  = τy ./ S.τ0
    return û, v̂, ŵ, p̂, b̂, τ̂x, τ̂y
end

"""
    redimensionalize_state(û, v̂, ŵ, p̂, b̂, τ̂x, τ̂y, S)

Convert nondimensional state fields back to dimensional form.

Returns `(u, v, w, p, b, τx, τy)`.
"""
function redimensionalize_state(û, v̂, ŵ, p̂, b̂, τ̂x, τ̂y, S::OceanScales)
    u   = û .* S.U0
    v   = v̂ .* S.U0
    w   = ŵ .* ((S.H0 / S.L) * S.U0)
    p   = p̂ .* S.p0
    b   = b̂ .* S.b0
    τx  = τ̂x .* S.τ0
    τy  = τ̂y .* S.τ0
    return u, v, w, p, b, τx, τy
end

# -----------------------------------------------------------------------------
# Bathymetry and buoyancy
# -----------------------------------------------------------------------------

"""
    nondimensionalize_bathymetry(H, S)

Return nondimensional bathymetry `Ĥ = H / H0`.
"""
nondimensionalize_bathymetry(H, S::OceanScales) = H ./ S.H0

"""
    redimensionalize_bathymetry(Ĥ, S)

Return dimensional bathymetry `H = Ĥ * H0`.
"""
redimensionalize_bathymetry(Ĥ, S::OceanScales) = Ĥ .* S.H0

"""
    nondimensionalize_buoyancy(b, S)

Return nondimensional buoyancy `b̂ = b / b0`.
"""
nondimensionalize_buoyancy(b, S::OceanScales) = b ./ S.b0

"""
    redimensionalize_buoyancy(b̂, S)

Return dimensional buoyancy `b = b̂ * b0`.
"""
redimensionalize_buoyancy(b̂, S::OceanScales) = b̂ .* S.b0

end # module