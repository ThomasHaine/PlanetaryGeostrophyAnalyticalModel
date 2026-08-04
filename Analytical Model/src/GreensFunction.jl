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
    Gb_split(z, ξ, H, ψ)

Return coefficients (a+, a-) such that:

- for ξ <= z:  G_b = a+ * exp(ψ ξ) + a- * exp(-ψ ξ)
- for ξ >= z:  G_b = a+ * exp(ψ ξ) + a- * exp(-ψ ξ)

The coefficients depend on z and on the branch.
"""
function Gb_split_coeffs(z, H, ψ, branch::Symbol)
    denom = 2 * ψ * (1 + exp(2ψ*H))

    if branch == :lt
        # ξ <= z
        a_plus  = exp(ψ*z) * (1 - exp(-2ψ*z)) * exp(2ψ*H) / denom
        a_minus = exp(ψ*z) * (1 - exp(-2ψ*z)) / denom
    elseif branch == :gt
        # ξ >= z
        a_plus  = exp(ψ*z) * (exp(2ψ*H) + exp(-2ψ*z)) / denom
        a_minus = exp(ψ*z) * (exp(2ψ*H) + exp(-2ψ*z)) / denom
    else
        error("branch must be :lt or :gt")
    end

    return a_plus, a_minus
end