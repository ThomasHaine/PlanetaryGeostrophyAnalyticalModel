"""
    Gu(z, ξ, φ, H)

Vertical velocity Green's function for uniform viscosity.
All variables assumed nondimensional.
"""
function Gu(z, ξ, φ, H)
    denom = 2 * φ * (1 + exp(2 * φ * H))
    pref = exp(-φ * (z + ξ)) / denom

    if z <= ξ
        return pref * (exp(2φ * ξ) - exp(2φ * (z + H)) - exp(2φ * (z + ξ + H)) + 1)
    else
        return pref * (exp(2φ * z) - exp(2φ * (ξ + H)) - exp(2φ * (z + ξ + H)) + 1)
    end
end

"""
    Gb_coeffs(z, H, ψ, κ, branch)

Vertical buoyancy Green's function for uniform diffusivity.
All variables assumed nondimensional.

Return coefficients (A, B) such that
    G_b(z; ξ) = A * exp(ψ ξ) + B * exp(-ψ ξ)

for the chosen branch:
- branch = :lt  means ξ <= z
- branch = :gt  means ξ >= z
"""
function Gb_coeffs(z, H, ψ, κ, branch::Symbol)
    denom = 2 * κ * ψ * (1 + exp(2 * ψ * H))

    if branch == :lt
        # ξ <= z
        A = (1 + exp(2 * ψ * (z + H))) * exp(-ψ * z) / denom
        B = -A
    elseif branch == :gt
        # ξ >= z
        A = (exp(2 * ψ * z) - 1) * exp(ψ * (2H - z)) / denom
        B = (exp(ψ * z) - exp(-ψ * z)) / denom
    else
        error("branch must be :lt or :gt")
    end

    return A, B
end

# -----------------------------------------------------------------------------
# Exact integral of cubic polynomial times exponential
# -----------------------------------------------------------------------------

"""
    polyexp_int(c, α, a, b)

Compute exactly

    ∫ₐᵇ (c₀ + c₁ξ + c₂ξ² + c₃ξ³) * exp(α ξ) dξ

where `c = (c0, c1, c2, c3)`.
"""
function polyexp_int(c::NTuple{4,<:Real}, α::Real, a::Real, b::Real)
    @assert α != 0
    c0, c1, c2, c3 = c

    F0(x) = exp(α*x) / α
    F1(x) = exp(α*x) * (x/α - 1/α^2)
    F2(x) = exp(α*x) * (x^2/α - 2x/α^2 + 2/α^3)
    F3(x) = exp(α*x) * (x^3/α - 3x^2/α^2 + 6x/α^3 - 6/α^4)

    return (c0*F0(b) + c1*F1(b) + c2*F2(b) + c3*F3(b)) -
           (c0*F0(a) + c1*F1(a) + c2*F2(a) + c3*F3(a))
end

function interval_contrib(c::NTuple{4,Real}, a::Real, b::Real, z::Real, H::Real, ψ::Real, κ::Real)
    if b <= z
        A, B = Gb_coeffs(z, H, ψ, κ, :lt)
        return A * polyexp_int(c,  ψ, a, b) +
               B * polyexp_int(c, -ψ, a, b)

    elseif a >= z
        A, B = Gb_coeffs(z, H, ψ, κ, :gt)
        return A * polyexp_int(c,  ψ, a, b) +
               B * polyexp_int(c, -ψ, a, b)

    else
        return interval_contrib(c, a, z, z, H, ψ, κ) +
               interval_contrib(c, z, b, z, H, ψ, κ)
    end
end


# ----------------------------------------------------------------------------
# Derivative of G_b at ξ = 0
# ----------------------------------------------------------------------------

"""
    dGb_dξ_at_zero(z, H, ψ, κ)

Return ∂_ξ G_b(z; 0).
"""
function dGb_dξ_at_zero(z, H, ψ, κ)
    denom = κ * (1 + exp(2 * ψ * H))
    numer = exp(ψ*z)*(exp(2*ψ*H) + exp(-2*ψ*z))
    return numer/denom
end