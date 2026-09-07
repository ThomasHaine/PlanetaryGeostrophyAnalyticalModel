# Script to test the output of the ComputeQuadratures.ipynb notebook. The Symbolics.jl form of the double integral is loaded and then compared to numerical estimates of the quadrature for several parameters.
# TWNH Aug '26

using Symbolics
using JLD2
using Latexify
using QuadGK
using Printf
using Statistics
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

@variables x y z ξ ξp
@variables H
@variables Ek κ ν ψ φ f β

@load "symexpr.jld2" symbolics_Guvb symbolics_Gb symbolics_Guv symbolics_inner_Gb_integral symbolics_Guvb_xi symbolics_inner_Gb2_integral symbolics_Guv_y 
# symbolics_Guv_y_b_xi

Guvb_expr_fun = eval(build_function(symbolics_Guvb, H, ψ, ξp, φ, z, κ, ν; expression = Val(false)))
Guvb(H, z, ξp, φ, ψ, κ, ν) = Guvb_expr_fun(H, ψ, ξp, φ, z, κ, ν)

Guv_expr_fun = eval(build_function(symbolics_Guv, H, z, ξ, φ, ν; expression = Val(true)))
Guv_sym(z, ξ, H, ν, φ) = Guv_expr_fun(H, z, ξ, φ, ν)

Gb_expr_fun  = eval(build_function(symbolics_Gb,  H, z, ξ, ψ, κ; expression = Val(true)))
Gb_sym(z, ξ, H, κ, ψ) = Gb_expr_fun(H, z, ξ, ψ, κ)

inner_Gb_integral_expr_fun  = eval(build_function(symbolics_inner_Gb_integral, H, ξ, ξp, ψ, κ; expression = Val(true)))
inner_Gb_integral_sym(ξ, ξp, H, κ, ψ) = inner_Gb_integral_expr_fun(H, ξ, ξp, ψ, κ)

inner_Gb2_integral_expr_fun  = eval(build_function(symbolics_inner_Gb2_integral, H, ξ, ψ, κ; expression = Val(true)))
inner_Gb2_integral_sym(ξ, H, κ, ψ) = inner_Gb2_integral_expr_fun(H, ξ, ψ, κ)

Guvb_xi_expr_fun = eval(build_function(symbolics_Guvb_xi, H, z, φ, ψ, κ, ν; expression = Val(false)))
Guvb_xi(H, z, φ, ψ, κ, ν) = Guvb_xi_expr_fun(H, z, φ, ψ, κ, ν) 

Guv_y_expr_fun = eval(build_function(symbolics_Guv_y, H, z, ξ, φ, ν, β, f; expression = Val(false)))
Guv_y(H, z, ξ, φ, ν, β, f) = Guv_y_expr_fun(H, z, ξ, φ, ν, β, f) 

# Guv_y_b_xi_expr_fun = eval(build_function(symbolics_Guv_y_b_xi, H, z, φ, ψ, κ, ν, β, f; expression = Val(false)))
# Guv_y_b_xi(H, z, φ, ψ, κ, ν, β, f) = Guv_y_b_xi_expr_fun(H, z, φ, ψ, κ, ν, β, f) 

println("Finished generating symbolic functions.")
println()

# -------------------------------------------------------------------
# Numerical kernels from screenshots
# -------------------------------------------------------------------
function Guv_num(z, ξ, H, ν, φ)
    pref = exp(-φ*(z + ξ)) / (2 * ν * φ * (1 + exp(2 * φ * H)))

    if z <= ξ
        return pref * (exp(2*φ*ξ) - exp(2*φ*(z + H)) - exp(2*φ*(z + ξ + H)) + 1)
    else
        return pref * (exp(2*φ*z) - exp(2*φ*(ξ + H)) - exp(2*φ*(z + ξ + H)) + 1)
    end
end

function Gb_num(z, ξ, H, κ, ψ)
    pref = exp(ψ*(z + ξ)) / (2 * κ * ψ * (1 + exp(2 * ψ * H)))

    if z <= ξ
        return pref * (1 - exp(-2*ψ*ξ)) * (exp(2*ψ*H) + exp(-2*ψ*z))
    else
        return pref * (1 - exp(-2*ψ*z)) * (exp(2*ψ*H) + exp(-2*ψ*ξ))
    end
end

# -------------------------------------------------------------------
# Integrands and double integrals
# -------------------------------------------------------------------

function inner_Gb_integral_num(ξ, ξp, H, κ, ψ)
    quadgk(zp -> Gb_num(zp, ξp, H, κ, ψ), ξ, 0.0; rtol=1e-10, atol=1e-12)[1]
end

function inner_Gb_integral_num_split(ξ, ξp, H, κ, ψ; rtol=1e-11, atol=1e-13)
    f(zp) = Gb_num(zp, ξp, H, κ, ψ)

    # Integration interval is [ξ, 0].
    pts = [ξ, 0.0]

    # Split at zp = ξp if it lies strictly inside [ξ, 0].
    if ξ < ξp < 0.0
        push!(pts, ξp)
    end

    pts = sort(unique(pts))

    total = zero(complex(float(1.0)))
    for i in 1:length(pts)-1
        a = pts[i]
        b = pts[i+1]
        if a != b
            total += quadgk(f, a, b; rtol=rtol, atol=atol)[1]
        end
    end

    return total
end

function Guv_y_num(H, z, ξ, φ, ν, β, f; h=1e-7 * im)
    return ((β*φ)/(2 * f) * (Guv_num(z, ξ, H, ν, φ + h) - Guv_num(z, ξ, H, ν, φ - h)) / (2h))
end

function integrand_num(ξ, H, z, ξp, φ, ψ, κ, ν)
    Guv_num(z, ξ, H, ν, φ) * inner_Gb_integral_num(ξ, ξp, H, κ, ψ)
end

function integrand2_num(ξ, H, z, φ, ψ, κ, ν)
    Guv_num(z, ξ, H, ν, φ) * inner_Gb2_integral_sym(ξ, H, κ, ψ)
end

function integrand3_num(ξ, H, z, φ, ψ, κ, ν, βval, fval)
    Guv_y_num(H, z, ξ, φ, ν, βval, fval) * inner_Gb2_integral_sym(ξ, H, κ, ψ)
end

function integrand_num_split_inner(ξ, H, z, ξp, φ, ψ, κ, ν)
    Guv_num(z, ξ, H, ν, φ) * inner_Gb_integral_num_split(ξ, ξp, H, κ, ψ)
end

function double_integral_num_split_robust(H, z, ξp, φ, ψ, κ, ν; rtol=1e-11, atol=1e-13)
    f(ξ) = integrand_num_split_inner(ξ, H, z, ξp, φ, ψ, κ, ν)

    pts = [-H, 0.0]

    if -H < z < 0.0
        push!(pts, z)
    end

    if -H < ξp < 0.0
        push!(pts, ξp)
    end

    pts = sort(unique(pts))

    total = zero(complex(float(1.0)))
    for i in 1:length(pts)-1
        a = pts[i]
        b = pts[i+1]
        if a != b
            total += quadgk(f, a, b; rtol=rtol, atol=atol)[1]
        end
    end

    return total
end

function double_integral_num(H, z, ξp, φ, ψ, κ, ν)
    quadgk(ξ -> integrand_num(ξ, H, z, ξp, φ, ψ, κ, ν),
           -H, 0.0; rtol=1e-10, atol=1e-12)[1]
end

function double_integral2_num(H, z, φ, ψ, κ, ν)
    quadgk(ξ -> integrand2_num(ξ, H, z, φ, ψ, κ, ν),
           -H, 0.0; rtol=1e-10, atol=1e-12)[1]
end

function double_integral3_num(H, z, φ, ψ, κ, ν, βval, fval)
    quadgk(ξ -> integrand3_num(ξ, H, z, φ, ψ, κ, ν, βval, fval),
           -H, 0.0; rtol=1e-10, atol=1e-12)[1]
end

function double_integral_num_split(H, z, ξp, φ, ψ, κ, ν)
    f(ξ) = integrand_num(ξ, H, z, ξp, φ, ψ, κ, ν)

    # Candidate breakpoints
    pts = [-H, z, ξp, 0.0]

    # Keep only unique points strictly inside the interval
    pts = sort(unique(filter(t -> (-H < t < 0.0), pts)))

    # Build subinterval endpoints
    edges = vcat(-H, pts, 0.0)

    # Integrate each subinterval separately
    total = zero(complex(float(1.0)))  # ComplexF64 accumulator
    for i in 1:length(edges)-1
        a = edges[i]
        b = edges[i+1]
        if a != b
            total += quadgk(f, a, b; rtol=1e-12, atol=1e-14)[1]
        end
    end

    return total
end

function double_integral_sym_from_Green(H, z, ξp, φ, ψ, κ, ν)
    f(ξ) = Guv_sym(z, ξ, H, ν, φ) * inner_Gb_integral_sym(ξ, ξp, H, κ, ψ)

    pts = sort(unique(filter(t -> (-H < t < 0.0), [z, ξp])))
    edges = vcat(-H, pts, 0.0)

    total = zero(complex(float(1.0)))
    for i in 1:length(edges)-1
        total += quadgk(f, edges[i], edges[i+1]; rtol=1e-12, atol=1e-14)[1]
    end
    return total
end

# ------------------------------------------------------------
# Comparison helpers
# ------------------------------------------------------------

function compare_symbolic_vs_numeric(Guvb, Hval, zval, ξpval, φval, ψval, κval, νval)
    @time begin
        println("Computing double integral with Symbolics expression")
        sym_val = Guvb(Hval, zval, ξpval, φval, ψval, κval, νval)
    end
    @time begin
        println("Computing double integral with quadgk")
        num_val = double_integral_num(Hval, zval, ξpval, φval, ψval, κval, νval)
    end

    abs_err = abs(sym_val - num_val)
    rel_err = abs_err / max(abs(num_val), eps(Float64))

    return sym_val, num_val, abs_err, rel_err
end

"""
    match_ok(sym, num; rtol=1e-8, atol=1e-10)

Return true if `sym` and `num` agree within absolute/relative tolerances.
"""
function match_ok(sym, num; rtol=1e-8, atol=1e-10)
    return isapprox(sym, num; rtol=rtol, atol=atol)
end

"""
    report_match(name, sym, num; rtol=1e-8, atol=1e-10)

Print a one-line PASS/FAIL report for a symbolic vs numerical comparison.
"""
function report_match(name, sym, num; rtol=1e-8, atol=1e-10)
    ok = match_ok(sym, num; rtol=rtol, atol=atol)
    abs_err = abs(sym - num)
    rel_err = abs_err / max(abs(num), eps(real(num)))

    status = ok ? "PASS" : "FAIL"
    @printf("%-24s %-4s  abs=% .3e  rel=% .3e\n", name, status, abs_err, rel_err)

    return ok, abs_err, rel_err
end

# Example values
Hval  = 1.0
zval  = -0.6
ξpval = -0.5
φval  = 0.05 * sqrt(im)
ψval  = 0.05
νval  = 1.2
κval  = 1.1

zvals = [-0.99, -0.66, -0.33, -0.01]
ξvals = zvals
ξpvals = ξvals

# First test the integrand values:
function compare_Gu(z, ξ, H, ν, φ)
    sym = Guv_sym(z, ξ, H, ν, φ)
    num = Guv_num(z, ξ, H, ν, φ)
    abs_err = abs(sym - num)
    rel_err = abs_err / max(abs(num), eps(real(num)))
    return sym, num, abs_err, rel_err
end

function compare_Gb(z, ξ, H, κ, ψ)
    sym = Gb_sym(z, ξ, H, κ, ψ)
    num = Gb_num(z, ξ, H, κ, ψ)
    abs_err = abs(sym - num)
    rel_err = abs_err / max(abs(num), eps(real(num)))
    return sym, num, abs_err, rel_err
end

println("========================================================================================================================")
println("1.1 Testing Gu from ComputeQuadratures.ipynb and numerical Gu defined here:")
for z in zvals, ξ in ξvals
    sym, num, ae, re = compare_Gu(z, ξ, Hval, νval, φval)
    report_match("Guv(z=$z, ξ=$ξ)", sym, num; rtol=1e-10, atol=1e-12)
end

println()
println("========================================================================================================================")
println("1.2 Testing Gb from ComputeQuadratures.ipynb and numerical Gb defined here:")
for z in zvals, ξ in ξvals
    sym, num, ae, re = compare_Gb(z, ξ, Hval, κval, ψval)
    report_match("Gb(z=$z, ξ=$ξ)", sym, num; rtol=1e-10, atol=1e-12)
end

# Second test inner_b integral:
function inner_b_sym(ξ, ξp, H, κ, ψ)
    quadgk(zp -> Gb_sym(zp, ξp, H, κ, ψ), ξ, 0.0; rtol=1e-10, atol=1e-12)[1]
end

function compare_inner_b(ξ, ξp, H, κ, ψ)
    sym = inner_b_sym(ξ, ξp, H, κ, ψ)
    num = inner_Gb_integral_num(ξ, ξp, H, κ, ψ)
    sym2 = inner_Gb_integral_sym(ξ, ξp, H, κ, ψ)
    return sym, num, sym2
end

println()
println("========================================================================================================================")
println("1.3 Testing inner_b:")
for ξ in ξvals, ξp in ξpvals
    sym, num, sym2 = compare_inner_b(ξ, ξp, Hval, κval, ψval)
    report_match("inner_b(ξ=$ξ, ξp=$ξp)", sym, num; rtol=1e-10, atol=1e-12)
end

# Third test the integral
function integrand_sym(ξ, H, z, ξp, φ, ψ, κ, ν)
    Guv_sym(z, ξ, H, ν, φ) * inner_b_sym(ξ, ξp, H, κ, ψ)
end
function compare_outer_integrand(ξ, H, z, ξp, φ, ψ, κ, ν)
    sym = integrand_sym(ξ, H, z, ξp, φ, ψ, κ, ν)
    num = integrand_num(ξ, H, z, ξp, φ, ψ, κ, ν)
    abs_err = abs(sym - num)
    rel_err = abs_err / max(abs(num), eps(real(num)))
    return sym, num, abs_err, rel_err
end

println()
println("1.4 Testing outer integrand:")
for ξtest in [-0.9, -0.75, -0.5, -0.25, -0.1]
    sym, num, ae, re = compare_outer_integrand(ξtest, Hval, zval, ξpval, φval, ψval, κval, νval)
    report_match("outer_integrand(ξ=$ξtest)", sym, num; rtol=1e-10, atol=1e-12)
end

# Fourth test the double integral:
println()
println("========================================================================================================================")
println("1.5 Testing Guvb double integral")
num_val = double_integral_num(Hval, zval, ξpval, φval, ψval, κval, νval)
num_val_split = double_integral_num_split(Hval, zval, ξpval, φval, ψval, κval, νval)
sym_val = Guvb(Hval, zval, ξpval, φval, ψval, κval, νval)
sym_quad   = double_integral_sym_from_Green(Hval, zval, ξpval, φval, ψval, κval, νval)

println("           sym = ", sym_val)
println(" quad from sym = ", sym_quad)
println("           num = ", num_val)
println("     split num = ", num_val_split)

# More comprehensive tests for several parameters:
Ekvals = [1e-1, 1e-2, 1e-3, 1e-4]

function varphi_from_Ek(Ek; f = 1.0, ν = 1.0)
    return sqrt(complex(0,1) * f / (ν * Ek))
end

function psi_from_Ek(Ek; Γ = 1.0, κ = 1.0)
    return sqrt(Γ / (κ * Ek))
end

function benchmark_one(Ekval, Hval, zval, ξpval; νval=1.0, κval=1.0)
    φval = varphi_from_Ek(Ekval; ν = νval)
    ψval = psi_from_Ek(Ekval; κ = κval)

    t_sym = @elapsed sym_val = Guvb(Hval, zval, ξpval, φval, ψval, κval, νval)

    t_num = NaN
    num_val = NaN + NaN*im
    ok = true
    try
        t_num = @elapsed num_val = double_integral_num_split_robust(
            Hval, zval, ξpval, φval, ψval, κval, νval
        )

    catch err
        ok = false
        @warn "quadgk failed" Ekval zval err
    end

    if ok
        abs_err = abs(sym_val - num_val)
        rel_err = abs_err / max(abs(num_val), eps(real(num_val)))
    else
        abs_err = NaN
        rel_err = NaN
    end

    return (Ek=Ekval, z=zval, ξp=ξpval , φ=φval, ψ=ψval,
            sym=sym_val, num=num_val, abs_err=abs_err, rel_err=rel_err,
            t_sym=t_sym, t_num=t_num, ok=ok)
end

function summarize_results(results)
    good = filter(r -> r.ok, results)

    if isempty(good)
        println("No successful runs.")
        return
    end

    t_sym_mean = mean(r.t_sym for r in good)
    t_num_mean = mean(r.t_num for r in good)
    rel_mean   = mean(r.rel_err for r in good)
    rel_max    = maximum(r.rel_err for r in good)

    println("\nOverall performance summary")
    println("------------------------------------------------")
    @printf("successful runs: %d / %d\n", length(good), length(results))
    @printf("mean symbolic time : %.3e s\n", t_sym_mean)
    @printf("mean quadgk time   : %.3e s\n", t_num_mean)
    @printf("mean rel error     : %.3e\n", rel_mean)
    @printf("max  rel error     : %.3e\n", rel_max)
end

results = Vector{Any}()

for Ekval in Ekvals
    for zval in zvals
        for ξpval in ξpvals
            push!(results, benchmark_one(Ekval, Hval, zval, ξpval; νval=νval, κval=κval))
        end
    end
end

summarize_results(results)

println("\nPer-case summary:")
for r in results
    @printf("Ek=%-8.1e z=%-8.4f ξp=%-8.4f ok=%-5s  abserr=%-10.3e  relerr=%-10.3e  |num|=%-10.3e  t_sym=%-9.3e  t_num=%-9.3e\n",
        r.Ek, r.z, r.ξp, string(r.ok),
        r.abs_err, r.rel_err, abs(r.num), r.t_sym, r.t_num)
end

println()
println("========================================================================================================================")
println("1.6 Testing Guv_y derivative")
fval  = 1.0
βval  = 0.1
num2_val = Guv_y_num(Hval, zval, ξpval, φval, νval, βval, fval)
sym2_val = Guv_y(    Hval, zval, ξpval, φval, νval, βval, fval)

println("           sym = ", sym2_val)
println("           num = ", num2_val)


println()
println("========================================================================================================================")
println("1.7 Testing Guvb_xi double integral")
num3_val = double_integral2_num(Hval, zval, φval, ψval, κval, νval)
sym3_val = Guvb_xi(Hval, zval, φval, ψval, κval, νval)

println("           sym = ", sym3_val)
println("           num = ", num3_val)


# println()
# println("========================================================================================================================")
# println("1.7 Testing Guv_y_b_xi double integral")


# num4_val = double_integral3_num(Hval, zval, φval, ψval, κval, νval, βval, fval)
# sym4_val = Guv_y_b_xi(Hval, zval, φval, ψval, κval, νval, βval, fval)

# println("           sym = ", sym4_val)
# println("           num = ", num4_val)