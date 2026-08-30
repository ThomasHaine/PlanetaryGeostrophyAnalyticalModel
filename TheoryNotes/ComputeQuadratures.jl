#   Julia notebook to compute the symbolic quadratures for the solution of
#   the robust diagnostic PG problem
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   twnh 
# 
#   Aug '26. 
# 
#   This notebook reads the output from BuoyancyEquationSolution.ipynb and
#   FrictionalGeostrophicEquationSolution.ipynb, namely the Green's functions
#   G_{\mathfrak{u}} and G_b and computes integrals using them. It then writes
#   out the following expression for later use.
# 
#   \begin{align} G{ \mathfrak{u}, b} (z; \xi') & = \int{-H}^{0} G{
#   \mathfrak{u}} (z; \xi) \, \int{\xi}^{0} G_b(z'; \xi') \; dz' \; d \xi .
#   \end{align}
# 
#   See SymbolicsTesting.jl to test this result against numerical quadrature.
# 
#   The notebook makes a plot of this G_{ \mathfrak{u}, b} (z; \xi') function.

using SymPy
using Symbolics
using JLD2

# Geometry symbolic parameters:
z, ξ   = symbols("z ξ",   real=true, negative=true) # Vertical coordinate and source location (both in [-H,0])
x, y   = symbols("x y",   real=true)                # Horizontal coordinates
H      = symbols("H",   real=true, positive=true)   # Domain depth

ψ      = symbols("ψ",     real=true, positive=true) 
φ      = symbols("φ", imaginary=true)

κ      = symbols("κ",     real=true, positive=true)  # Diffusivity parameter
ν      = symbols("ν",     real=true, positive=true)  # Viscosity parameters

# Dictionary for parsing strings.
locals = Dict(
    "x" => x,
    "y" => y,
    "z" => z,
    "ξ" => ξ,
    "κ" => κ,
    "ν" => ν,
    "φ" => φ,
    "ψ" => ψ,
    "H" => H
    )

# Parse expressions using the canonical symbols.
Guv_str = read("sympy_Guv_expression.txt", String)
Guv = sympy.sympify(Guv_str, locals=locals)
println("Guv expression loaded from file:")
display(Guv)
println()

Gb_str  = read("sympy_Gb_expression.txt", String)
Gb  = sympy.sympify(Gb_str,  locals=locals)
println("Gb expression loaded from file:")
display(Gb)
println()

@time begin
    zp = symbols("zp",     real=true, negative=true)
    ξp = symbols("ξp",     real=true, negative=true)
    tmp_Gb = sympy.expand(Gb.subs(Dict(z => zp, ξ => ξp)))
    inner_Gb_integral = sympy.integrate(tmp_Gb, (zp, ξ, 0))
    max_obj = sympy.Max(ξ,ξp)
    min_obj = sympy.Min(0, max_obj)
    inner_Gb_integral = inner_Gb_integral.subs(min_obj,max_obj)
    # Create two explicit branches
    expr_if_xi_ge_xip = inner_Gb_integral.subs(sympy.Max(ξ, ξp), ξ)
    expr_if_xi_lt_xip = inner_Gb_integral.subs(sympy.Max(ξ, ξp),ξp)

    # Form a piecewise object
    inner_Gb_integral = sympy.Piecewise(
    (expr_if_xi_ge_xip, sympy.Ge(ξ, ξp)),
    (expr_if_xi_lt_xip, sympy.Lt(ξ, ξp)))
end

println("1.1 Symbolic computation of the integral of Gb completed.")
display(inner_Gb_integral)

@time begin
    # ------------------------------------------------------------------
    # Extract branch expressions
    #
    # Guv:
    #   Gu_branch1: z ≤ ξ  ⇔ ξ ≥ z
    #   Gu_branch2: z ≥ ξ  ⇔ ξ ≤ z
    #
    # inner_Gb_integral:
    #   Gbint_branch1: ξ ≥ ξp
    #   Gbint_branch2: ξ <  ξp
    # ------------------------------------------------------------------
    Gu_branch1 = Guv.args[1].args[1]                    # z ≤ ξ
    Gu_branch2 = Guv.args[2].args[1]                    # z ≥ ξ

    Gbint_branch1 = inner_Gb_integral.args[1].args[1]   # ξ ≥ ξp
    Gbint_branch2 = inner_Gb_integral.args[2].args[1]   # ξ <  ξp

    # ------------------------------------------------------------------
    # Products
    # ------------------------------------------------------------------
    f11 = sympy.simplify(Gu_branch1 * Gbint_branch1)  # ξ ≥ z, ξ ≥ ξp
    f12 = sympy.simplify(Gu_branch1 * Gbint_branch2)  # ξ ≥ z, ξ <  ξp
    f21 = sympy.simplify(Gu_branch2 * Gbint_branch1)  # ξ ≤ z, ξ ≥ ξp
    f22 = sympy.simplify(Gu_branch2 * Gbint_branch2)  # ξ ≤ z, ξ <  ξp

    # ------------------------------------------------------------------
    # Case A: z ≤ ξp
    #
    # Ordering:
    #
    #   -H ≤ ... ≤ z ≤ ξp ≤ ... ≤ 0
    #
    # Therefore:
    #
    #   ξ ∈ [-H, z]     → ξ ≤ z and ξ < ξp   → f22
    #   ξ ∈ [z, ξp]     → ξ ≥ z and ξ < ξp   → f12
    #   ξ ∈ [ξp, 0]     → ξ ≥ z and ξ ≥ ξp   → f11
    # ------------------------------------------------------------------
    I_A_22 = sympy.integrate(f22, (ξ, -H, z))
    I_A_12 = sympy.integrate(f12, (ξ,  z, ξp))
    I_A_11 = sympy.integrate(f11, (ξ, ξp, 0))

    I_A = sympy.simplify(I_A_22 + I_A_12 + I_A_11)

    # ------------------------------------------------------------------
    # Case B: ξp < z
    #
    # Ordering:
    #
    #   -H ≤ ... ≤ ξp < z ≤ ... ≤ 0
    #
    # Therefore:
    #
    #   ξ ∈ [-H, ξp]    → ξ < ξp and ξ ≤ z   → f22
    #   ξ ∈ [ξp, z]     → ξ ≥ ξp and ξ ≤ z   → f21
    #   ξ ∈ [z, 0]      → ξ ≥ ξp and ξ ≥ z   → f11
    # ------------------------------------------------------------------
    I_B_22 = sympy.integrate(f22, (ξ, -H, ξp))
    I_B_21 = sympy.integrate(f21, (ξ, ξp, z))
    I_B_11 = sympy.integrate(f11, (ξ,  z, 0))

    I_B = sympy.simplify(I_B_22 + I_B_21 + I_B_11)

    # ------------------------------------------------------------------
    # Final piecewise outer integral
    # ------------------------------------------------------------------
    Guvb = sympy.Piecewise(
        (I_A, sympy.Le(z, ξp)),
        (I_B, sympy.Gt(z, ξp))
    )
end

println("1.2 Symbolic computation of the integral of G_u and Gb completed.")
display(Guvb)

function sympy_piecewise_to_symbolics(P, condsym)
    branches = P.args
    expr1 = sympy_to_symbolics(branches[1].args[1], [])
    expr2 = sympy_to_symbolics(branches[2].args[1], [])
    return ifelse(condsym, expr1, expr2)
end

@variables z ξ ξp
symbolics_Guvb = sympy_piecewise_to_symbolics(Guvb, z <= ξp)
symbolics_inner_Gb_integral = sympy_piecewise_to_symbolics(inner_Gb_integral, ξ >= ξp)
symbolics_Gb   = sympy_piecewise_to_symbolics(Gb, z <= ξ)
symbolics_Guv  = sympy_piecewise_to_symbolics(Guv, z <= ξ)

@save "symexpr.jld2" symbolics_Guvb symbolics_Gb symbolics_Guv symbolics_inner_Gb_integral
println("1.3 Expression converted to Symbolics and saved to file: symexpr.jld2")
