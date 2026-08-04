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