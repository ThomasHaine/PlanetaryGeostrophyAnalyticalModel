"""
    Gu(z, ξ, φ, H)

Vertical Green's function for uniform viscosity.
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