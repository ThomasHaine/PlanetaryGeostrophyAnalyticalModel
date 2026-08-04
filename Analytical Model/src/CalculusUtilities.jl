"""
    polyexp_int(c, α, a, b)

Compute the exact integral

    ∫ₐᵇ (c₀ + c₁ξ + c₂ξ² + c₃ξ³) * exp(α ξ) dξ

where `c = (c0, c1, c2, c3)`.

Assumes α ≠ 0.
"""
function polyexp_int(c::NTuple{4,Real}, α::Real, a::Real, b::Real)
    @assert α != 0

    c0, c1, c2, c3 = c

    # Antiderivatives evaluated at x:
    # ∫ e^{αx} dx
    F0(x) = exp(α*x) / α

    # ∫ x e^{αx} dx
    F1(x) = exp(α*x) * (x/α - 1/α^2)

    # ∫ x^2 e^{αx} dx
    F2(x) = exp(α*x) * (x^2/α - 2x/α^2 + 2/α^3)

    # ∫ x^3 e^{αx} dx
    F3(x) = exp(α*x) * (x^3/α - 3x^2/α^2 + 6x/α^3 - 6/α^4)

    Ia = c0*F0(a) + c1*F1(a) + c2*F2(a) + c3*F3(a)
    Ib = c0*F0(b) + c1*F1(b) + c2*F2(b) + c3*F3(b)

    return Ib - Ia
end