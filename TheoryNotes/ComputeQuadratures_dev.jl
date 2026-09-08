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
# 
#   The notebook also computes 
# 
#   \begin{align} G{ \mathfrak{u}, {b\xi}} (z) & = \int{-H}^{0} G{ \mathfrak{u}}
#   (z; \xi) \, \int{\xi}^{0} \partial\xi Gb(z';\xi) |{\xi = 0} \; dz' \; d \xi
#   ,\ G{ \mathfrak{u}y, b}& = \int{-H}^{0} \partialy G{ \mathfrak{u}} (z; \xi)
#   \, \int{\xi}^{0} Gb(z';\xi) \; dz' \; d \xi , \ G{ \mathfrak{u}y, {b\xi}}
#   (z) & = \int{-H}^{0} \partialy G{ \mathfrak{u}} (z; \xi) \, \int{\xi}^{0}
#   \partial\xi Gb(z';\xi) |_{\xi = 0} \; dz' \; d \xi . \end{align}

using SymPy
using Symbolics
using JLD2
using Infiltrator

# Geometry symbolic parameters:
z,  ξ  = symbols("z  ξ",   real=true, negative=true) # Vertical coordinate and source location (both in [-H,0])
zp, ξp = symbols("zp ξp",  real=true, negative=true) # Vertical coordinate and source location (both in [-H,0])
x, y   = symbols("x y",    real=true)                # Horizontal coordinates
H      = symbols("H",      real=true, positive=true) # Domain depth

ψ      = symbols("ψ",      real=true, positive=true) 
φ      = symbols("φ", imaginary=true)

κ      = symbols("κ",      real=true, positive=true) # Diffusivity parameter
ν      = symbols("ν",      real=true, positive=true) # Viscosity parameters

Ek     = symbols("Ek",     real=true, positive=true) # Ekman number
f      = symbols("f",      real=true)                # Coriolis parameter
β      = symbols("β",      real=true, positive=true) # Planetary vorticity gradient

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
println("0.1 Guv expression loaded from file:")
println()

Gb_str  = read("sympy_Gb_expression.txt", String)
Gb  = sympy.sympify(Gb_str,  locals=locals)
println("0.2 Gb expression loaded from file:")
println()

# ---------------------------------------------------------------------
# Faster symbolic utilities
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Faster symbolic utilities
# ---------------------------------------------------------------------

function contains_symbol(expr, var)
    try
        return var in expr.free_symbols
    catch
        return occursin(string(var), string(expr))
    end
end

function has_unevaluated_integral(expr)
    s = string(expr)
    return occursin("Integral", s)
end

function bad_integral_result(expr, var)
    return contains_symbol(expr, var) ||
           has_unevaluated_integral(expr) ||
           occursin("Piecewise", string(expr))
end

function light_simplify(expr)
    expr = sympy.powsimp(expr, force=false)
    expr = sympy.factor_terms(expr)
    expr = sympy.cancel(expr)
    return expr
end

function medium_simplify(expr)
    expr = sympy.together(expr)
    expr = sympy.cancel(expr)
    expr = sympy.factor_terms(expr)
    expr = sympy.powsimp(expr, force=false)
    return expr
end

const integral_cache = Dict{Tuple{String,String,String,String}, Any}()
const dep_integral_cache = Dict{Tuple{String,String,String,String}, Any}()

function sympy_is_add(expr)
    try
        return Bool(expr.is_Add)
    catch
        return occursin("Add", string(expr.func))
    end
end

function additive_args(expr)
    if sympy_is_add(expr)
        return collect(expr.args)
    else
        return Any[expr]
    end
end

function top_level_terms(expr; do_expand=true, split_rational_numerator=true)
    ex = do_expand ? sympy.expand_mul(expr) : expr

    # First try ordinary additive splitting.
    ex_expanded = do_expand ? sympy.expand(ex) : ex
    terms = additive_args(ex_expanded)

    if length(terms) > 1
        return terms
    end

    if !split_rational_numerator
        return terms
    end

    # Fallback:
    # If expr is one huge rational expression, split the numerator.
    num, den = sympy.fraction(ex_expanded)

    num_expanded = sympy.expand(num)
    num_terms = additive_args(num_expanded)

    if length(num_terms) > 1
        println("    split rational numerator into $(length(num_terms)) terms")
        return Any[t / den for t in num_terms]
    end

    # Second fallback: use together() form, then split numerator.
    ex2 = sympy.together(ex)
    num2, den2 = sympy.fraction(ex2)
    num2 = sympy.expand(num2)
    num2_terms = additive_args(num2)

    if length(num2_terms) > 1
        println("    split together() numerator into $(length(num2_terms)) terms")
        return Any[t / den2 for t in num2_terms]
    end

    return Any[ex_expanded]
end

function normalize_for_integration(expr)
    # Stronger than the per-term powsimp, but applied only to one small term.
    expr = sympy.powsimp(expr, force=true)
    expr = sympy.cancel(expr)
    expr = sympy.factor_terms(expr)
    expr = sympy.powsimp(expr, force=true)
    return expr
end

function independent_dependent_parts(t, var)
    try
        # First normalize/cancel exponentials like
        # exp(-2ξψ)/(exp(ξφ)*exp(-ξψ)) -> exp(-ξφ - ξψ)
        tt = normalize_for_integration(t)

        parts = tt.as_independent(var, as_Add=false)
        indep = parts[1]
        dep   = parts[2]

        dep = sympy.powsimp(dep, force=true)

        return indep, dep
    catch
        return sympy.S(1), t
    end
end

function is_sympy_exp(expr)
    try
        return string(expr.func) == "exp"
    catch
        return false
    end
end

function combine_exp_products(expr)
    try
        expr = sympy.powsimp(expr, force=true)
        expr = sympy.expand_power_exp(expr)
        expr = sympy.powsimp(expr, force=true)
        return expr
    catch
        return expr
    end
end

function try_integrate_simple_exp(dep, var, a, b)
    dep = combine_exp_products(dep)

    # Case 1:
    #   dep = exp(linear(var))
    try
        if is_sympy_exp(dep)
            arg = dep.args[1]
            λ = sympy.diff(arg, var)

            if !contains_symbol(λ, var)
                return (dep.subs(var, b) - dep.subs(var, a)) / λ
            end
        end
    catch
    end

    # Case 2:
    #   dep = product of exponentials, e.g.
    #   exp(-ξ*φ)*exp(-ξ*ψ)
    #
    # Try to force it into exp(linear expression).
    try
        dep2 = sympy.powsimp(dep, force=true)

        if is_sympy_exp(dep2)
            arg = dep2.args[1]
            λ = sympy.diff(arg, var)

            if !contains_symbol(λ, var)
                return (dep2.subs(var, b) - dep2.subs(var, a)) / λ
            end
        end
    catch
    end

    # Case 3:
    #   dep = polynomial(var) * exp(linear(var))
    try
        dep2 = sympy.powsimp(dep, force=true)
        factors = collect(dep2.as_ordered_factors())

        exp_factor = nothing
        other = sympy.S(1)

        for fac in factors
            fac2 = sympy.powsimp(fac, force=true)

            if is_sympy_exp(fac2)
                if exp_factor === nothing
                    exp_factor = fac2
                else
                    exp_factor = sympy.powsimp(exp_factor * fac2, force=true)
                end
            else
                other *= fac2
            end
        end

        if exp_factor !== nothing
            exp_factor = sympy.powsimp(exp_factor, force=true)

            if is_sympy_exp(exp_factor)
                arg = exp_factor.args[1]
                λ = sympy.diff(arg, var)

                if contains_symbol(λ, var)
                    return nothing
                end

                poly = sympy.Poly(other, var)
                deg = Int(poly.degree())

                if deg <= 2
                    F = sympy.integrate(other * exp_factor, var, conds="none")
                    return F.subs(var, b) - F.subs(var, a)
                end
            end
        end
    catch
    end

    return nothing
end

function integrate_dep_cached(dep, var, a, b)
    dep = combine_exp_products(dep)

    key = (string(dep), string(var), string(a), string(b))

    if haskey(dep_integral_cache, key)
        return dep_integral_cache[key]
    end

    # If dep is actually independent of var.
    if !contains_symbol(dep, var)
        res = dep * (b - a)
        dep_integral_cache[key] = res
        return res
    end

    # Manual exponential cases.
    manual = try_integrate_simple_exp(dep, var, a, b)
    if manual !== nothing
        dep_integral_cache[key] = manual
        return manual
    end

    # Try a few tiny-core variants.
    candidates = (
        dep,
        sympy.powsimp(dep, force=true),
        sympy.expand_power_exp(dep),
        sympy.powsimp(sympy.expand_power_exp(dep), force=true),
        sympy.cancel(sympy.powsimp(dep, force=true)),
    )

    for dep_c in candidates
        dep_c = combine_exp_products(dep_c)

        manual = try_integrate_simple_exp(dep_c, var, a, b)
        if manual !== nothing
            dep_integral_cache[key] = manual
            return manual
        end

        res = sympy.integrate(dep_c, (var, a, b), conds="none")

        if !bad_integral_result(res, var)
            dep_integral_cache[key] = res
            return res
        end

        F = sympy.integrate(dep_c, var, conds="none")
        res = F.subs(var, b) - F.subs(var, a)

        if !bad_integral_result(res, var)
            dep_integral_cache[key] = res
            return res
        end
    end

    println("WARNING: integrate_dep_cached failed.")
    println("  dep = ", dep)
    println("  combined dep = ", combine_exp_products(dep))

    res = sympy.integrate(dep, (var, a, b), conds="none")
    dep_integral_cache[key] = res
    return res
end

function integrate_term_cached(t, var, a, b)
    # Normalize full term first. This is important for cancelling exponentials
    # between numerator and denominator.
    t_norm = normalize_for_integration(t)

    key = (string(t_norm), string(var), string(a), string(b))

    if haskey(integral_cache, key)
        return integral_cache[key]
    end

    indep, dep = independent_dependent_parts(t_norm, var)

    dep = combine_exp_products(dep)

    dep_int = integrate_dep_cached(dep, var, a, b)
    res = indep * dep_int

    integral_cache[key] = res
    return res
end

function diagnose_integrand(expr, var; name="expr")
    println()
    println("Diagnostics for $name")
    println("  full func:          ", expr.func)
    println("  full is_Add:        ", sympy_is_add(expr))
    println("  full args length:   ", length(expr.args))
    println("  full string length: ", length(string(expr)))

    ex = sympy.expand_mul(expr)
    ex = sympy.expand(ex)

    println("  expanded func:          ", ex.func)
    println("  expanded is_Add:        ", sympy_is_add(ex))
    println("  expanded args length:   ", length(ex.args))
    println("  expanded string length: ", length(string(ex)))

    num, den = sympy.fraction(ex)
    num_exp = sympy.expand(num)
    den_exp = sympy.expand(den)

    println("  numerator func:          ", num_exp.func)
    println("  numerator is_Add:        ", sympy_is_add(num_exp))
    println("  numerator args length:   ", length(num_exp.args))
    println("  numerator string length: ", length(string(num_exp)))
    println("  denominator contains $var: ", contains_symbol(den_exp, var))
    println("  denominator string length: ", length(string(den_exp)))

    num_terms = additive_args(num_exp)
    println("  numerator additive terms: ", length(num_terms))
    println()
end

function integrate_terms_fast(expr, integ_tuple;
                              simplify_input=false,
                              compress_every=10_000,
                              max_single_term_chars=20_000,
                              print_each_term=true)

    var, a, b = integ_tuple

    if simplify_input
        expr = light_simplify(expr)
    end

    terms = top_level_terms(expr; do_expand=true, split_rational_numerator=true)

    res = sympy.S(0)

    nterms = length(terms)
    println("    integrating $nterms terms from $a to $b ...")

    if nterms == 1
        term_size = length(string(terms[1]))

        println("    WARNING: only one top-level term found.")
        println("    term head = ", terms[1].func)
        println("    term size = ", term_size)

        if term_size > max_single_term_chars
            diagnose_integrand(expr, var; name="failed one-term integrand")
            error("Refusing to integrate one huge symbolic term of size $term_size.")
        end
    end

    for (i, t) in enumerate(terms)
        if print_each_term
            println("      term $i / $nterms, chars = ", length(string(t)))
        end

        # Very light preprocessing only.
        t_s = sympy.powsimp(t, force=false)

        indep, dep = independent_dependent_parts(t_s, var)

        if print_each_term
            println("        indep chars = ", length(string(indep)),
                ", dep chars = ", length(string(dep)),
                ", dep = ", dep)
        end

        elapsed = @elapsed begin
            ti = integrate_term_cached(t_s, var, a, b)
            res += ti
        end

        if print_each_term
            println("        done in ", round(elapsed, digits=4), " s")
        elseif i % 10 == 0
            println("      integrated $i / $nterms")
        end

        if bad_integral_result(ti, var)
            println("WARNING: bad term integral at term $i / $nterms")
            println("  term   = ", t_s)
            println("  result = ", ti)
            @infiltrate
        end

        if i % compress_every == 0
            # Mild compression only.
            res = sympy.factor_terms(res)
            res = sympy.powsimp(res, force=false)
            println("      compressed accumulated result at $i / $nterms")
        end
    end

    println("    simplifying accumulated integral...")
    res = medium_simplify(res)

    if bad_integral_result(res, var)
        println("WARNING: final integrated result still contains $var or bad structure.")
        @infiltrate
    end

    return res
end

function diagnose_integrand(expr, var; name="expr")
    println()
    println("Diagnostics for $name")
    println("  full func:          ", expr.func)
    println("  full is_Add:        ", sympy_is_add(expr))
    println("  full args length:   ", length(expr.args))
    println("  full string length: ", length(string(expr)))

    ex = sympy.expand_mul(expr)
    ex = sympy.expand(ex)

    println("  expanded func:          ", ex.func)
    println("  expanded is_Add:        ", sympy_is_add(ex))
    println("  expanded args length:   ", length(ex.args))
    println("  expanded string length: ", length(string(ex)))

    num, den = sympy.fraction(ex)
    num_exp = sympy.expand(num)
    den_exp = sympy.expand(den)

    println("  numerator func:          ", num_exp.func)
    println("  numerator is_Add:        ", sympy_is_add(num_exp))
    println("  numerator args length:   ", length(num_exp.args))
    println("  numerator string length: ", length(string(num_exp)))
    println("  denominator contains $var: ", contains_symbol(den_exp, var))
    println("  denominator string length: ", length(string(den_exp)))

    num_terms = additive_args(num_exp)
    println("  numerator additive terms: ", length(num_terms))
    println()
end

println()
println("1.3 Symbolic computation of integral of Gb_ξ...")
@time begin
    tmp_Gb2 = sympy.diff(Gb,ξ).subs(ξ,0).subs(z,zp)
    inner_Gb2_integral = sympy.integrate(tmp_Gb2, (zp, ξ, 0))
end
println("1.3 Done.")

println()
println("1.5 Symbolic computation of Guv_φ and Guv_y...")
@time begin
    Guv_φ = sympy.diff(Guv,φ)
    fy    = SymFunction("f",  real=true)                # Coriolis parameter
    φ_term = symbols("φ_term", imaginary=true)
    φ_eqn = sqrt(im * fy(y) / (ν * Ek))
    φ_derivative = diff(φ_eqn,y).subs(diff(fy(y),y),β)
    Latex_φ_derivative = β/ (2 * fy(y)) * φ_term
    @assert Latex_φ_derivative.subs(φ_term, φ_eqn) == φ_derivative
    φ_derivative = Latex_φ_derivative.subs(fy(y), f).subs(φ_term, φ)
    Guv_y = φ_derivative * Guv_φ
    Guv_y = sympy.simplify(Guv_y)
    end
println("1.5 Done and Latex formula for φ derivative passes.")

println()
println("1.6 Symbolic computation of Guv_y_b_xi...")

@time begin
    empty!(integral_cache)
    empty!(dep_integral_cache)

    inner_Gb2_integral_s = medium_simplify(inner_Gb2_integral)
    println("  Simplified inner_Gb2_integral.")

    Guvφ_branch1 = Guv_φ.args[1].args[1]
    Guvφ_branch2 = Guv_φ.args[2].args[1]

    # Simplify the individual Green's-function branches only.
    Guvφ_branch1 = light_simplify(Guvφ_branch1)
    Guvφ_branch2 = light_simplify(Guvφ_branch2)

    println("  Prepared Guv_φ branches.")

    # Expand products into additive terms.
    # Do NOT cancel/factor the whole branch after this.
    branch1 = sympy.expand_mul(Guvφ_branch1 * inner_Gb2_integral_s)
    branch2 = sympy.expand_mul(Guvφ_branch2 * inner_Gb2_integral_s)

    n1 = length(top_level_terms(branch1; do_expand=true, split_rational_numerator=true))
    n2 = length(top_level_terms(branch2; do_expand=true, split_rational_numerator=true))

    println("  Prepared product branches.")
    println("    branch1 terms: ", n1)
    println("    branch2 terms: ", n2)

    # diagnose_integrand(branch1, ξ; name="branch1")
    # diagnose_integrand(branch2, ξ; name="branch2")

    I1 = integrate_terms_fast(branch1, (ξ, z, 0);
                              simplify_input=false,
                              compress_every=10_000,
                              print_each_term=false)

    println("  Computed I1 = ∫_{z}^{0} Guv_φ * Gb_ξ dξ")

    I2 = integrate_terms_fast(branch2, (ξ, -H, z);
                              simplify_input=false,
                              compress_every=10_000,
                              print_each_term=false)

    println("  Computed I2 = ∫_{-H}^{z} Guv_φ * Gb_ξ dξ")

    println("  Simplifying I1 and I2...")
    I1 = medium_simplify(I1)
    I2 = medium_simplify(I2)

    Guv_φ_b_xi = medium_simplify(I1 + I2)

    println("  Combined I1 + I2.")

    # Multiply by dφ/dy only after integration.
    Guv_y_b_xi = φ_derivative * Guv_φ_b_xi

    println("  Simplifying final Guv_y_b_xi...")
    Guv_y_b_xi = sympy.together(Guv_y_b_xi)
    Guv_y_b_xi = sympy.cancel(Guv_y_b_xi)
    Guv_y_b_xi = sympy.factor_terms(Guv_y_b_xi)
    Guv_y_b_xi = sympy.powsimp(Guv_y_b_xi, force=false)

    println("  Final simplification done.")
end

println("1.6 Done.")

println("1.7 Checking final Guv_y_b_xi...")

@time begin
    Guv_y_b_xi_str = string(Guv_y_b_xi)

    println("  string length: ", length(Guv_y_b_xi_str))
    println("  contains unevaluated Integral: ", occursin("Integral", Guv_y_b_xi_str))
    println("  contains Piecewise:             ", occursin("Piecewise", Guv_y_b_xi_str))
    println("  contains zoo:                   ", occursin("zoo", Guv_y_b_xi_str))
    println("  contains nan:                   ", occursin("nan", lowercase(Guv_y_b_xi_str)))
end

println("1.8 Saving SymPy string form...")

@time begin
    @save "Guv_y_b_xi_sympy_string.jld2" Guv_y_b_xi_str
end

println("1.8 Done.")

using Symbolics

println("1.9 Converting SymPy expression to Symbolics.jl...")

@time begin
    symbolics_Guv_y_b_xi = sympy_to_symbolics(Guv_y_b_xi, [])
end

@time begin
    @save "Guv_y_b_xi_symbolics.jld2" symbolics_Guv_y_b_xi
end

println("1.9 Done.")