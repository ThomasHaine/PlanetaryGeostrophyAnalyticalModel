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

function interval_contrib(c::NTuple{4,<:Real}, a::Real, b::Real, z::Real, H::Real, ψ::Real, κ::Real)
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

"""
    Gb_stable(z, ξ, H, ψ, κ)

Overflow-safe buoyancy Green's function for uniform diffusivity.

Assumes nondimensional variables and -H <= z, ξ <= 0.
"""
function Gb_stable(z, ξ, H, ψ, κ)
    @assert H > 0
    @assert ψ > 0
    @assert κ > 0

    # Small numerical tolerance because z, ξ may be grid values.
    tol = 100eps(Float64)
    @assert -H - tol <= z <= 0.0 + tol
    @assert -H - tol <= ξ <= 0.0 + tol

    # r = exp(-2ψH), always <= 1 and usually tiny.
    r = exp(-2 * ψ * H)

    denom = 2 * κ * ψ * (1 + r)

    if z <= ξ
        # Original branch:
        #
        # G = exp(ψ(z+ξ)) / [2κψ(1+exp(2ψH))]
        #     * (1 - exp(-2ψξ)) * (exp(2ψH) + exp(-2ψz))
        #
        # Divide by exp(2ψH):
        #
        # G = exp(ψ(z+ξ)) / [2κψ(1+r)]
        #     * (1 - exp(-2ψξ)) * (1 + exp(-2ψ(z+H)))
        #
        # Combine the first two factors to avoid overflow:
        #
        # exp(ψ(z+ξ)) * (1 - exp(-2ψξ))
        # = exp(ψ(z+ξ)) - exp(ψ(z-ξ)).
        #
        # Both exponents are <= 0 for z <= ξ <= 0.
        a = exp(ψ * (z + ξ)) - exp(ψ * (z - ξ))
        b = 1 + exp(-2 * ψ * (z + H))
        return a * b / denom
    else
        # Original branch:
        #
        # G = exp(ψ(z+ξ)) / [2κψ(1+exp(2ψH))]
        #     * (exp(-2ψz) - 1) * (exp(2ψH) + exp(-2ψξ))
        #
        # Divide by exp(2ψH):
        #
        # G = exp(ψ(z+ξ)) / [2κψ(1+r)]
        #     * (exp(-2ψz) - 1) * (1 + exp(-2ψ(ξ+H)))
        #
        # Combine:
        #
        # exp(ψ(z+ξ)) * (exp(-2ψz) - 1)
        # = exp(ψ(ξ-z)) - exp(ψ(z+ξ)).
        #
        # Both exponents are <= 0 for ξ <= z <= 0.
        a = exp(ψ * (ξ - z)) - exp(ψ * (z + ξ))
        b = 1 + exp(-2 * ψ * (ξ + H))
        return a * b / denom
    end
end

"""
    dGb_dξ_at_zero_stable(z, H, ψ, κ)

Overflow-safe ∂ξ G_b(z;ξ)|_{ξ=0}.
"""
function dGb_dξ_at_zero_stable(z, H, ψ, κ)
    @assert H > 0
    @assert ψ > 0
    @assert κ > 0

    r = exp(-2 * ψ * H)

    numer = exp(ψ * z) * (1 + exp(-2 * ψ * (z + H)))
    denom = κ * (1 + r)

    return numer / denom
end

const GL8_X = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
     0.1834346424956498,
     0.5255324099163290,
     0.7966664774136267,
     0.9602898564975363
)

const GL8_W = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763
)

"""
    Gb_times_spline_integral_quadrature(spl, z, H, ψ, κ)

Compute

    ∫_{-H}^0 G_b(z; ξ) B(ξ) dξ

using overflow-safe G_b and Gauss-Legendre quadrature over spline knot intervals.
"""
function Gb_times_spline_integral_quadrature(spl, z, H, ψ, κ)
    s = spl.spline
    knots = collect(s.basis.t)

    lo = -H
    hi = 0.0

    acc = 0.0

    @inbounds for k in 1:length(knots)-1
        a0 = knots[k]
        b0 = knots[k+1]

        # Skip repeated knots.
        b0 <= a0 && continue

        # Clip to water column.
        a = max(a0, lo)
        b = min(b0, hi)

        b <= a && continue

        mid = (a + b) / 2
        rad = (b - a) / 2

        for q in eachindex(GL8_X)
            ξ = mid + rad * GL8_X[q]
            acc += GL8_W[q] * Gb_stable(z, ξ, H, ψ, κ) * spl(ξ) * rad
        end
    end

    return acc
end