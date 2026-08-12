# Script to test the output of the ComputeQuadratures.ipynb notebook. The Symbolics.jl form of the double integral is loaded and then compared to numerical estimates of the quadrature for several parameters.
# TWNH Aug '26

using Symbolics
using JLD2
using Latexify
using QuadGK
using Printf
using Statistics

@variables x y z ξ ξp
@variables H
@variables Ek κ ν ψ φ

@load "symexpr.jld2" symbolics_expr
vars = [H, ψ, ξp, φ, z, κ, ν]
println("These should match!")
println(vars)
println(collect(Symbolics.get_variables(symbolics_expr)))
f_expr = build_function(symbolics_expr, vars; expression = Val(false))
f2 = eval(f_expr)
Guvb(H, z, ξp, φ, ψ, κ, ν) = f2([H, ψ, ξp, φ, z, κ, ν])

println("Finished generating symbolic function.")
println()

# -------------------------------------------------------------------
# Numerical kernels from screenshots
# -------------------------------------------------------------------
function Gu_num(z, ξ, H, ν, φ)
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

function inner_b(ξ, ξp, H, κ, ψ)
    quadgk(zp -> Gb_num(zp, ξp, H, κ, ψ), ξ, 0.0; rtol=1e-10, atol=1e-12)[1]
end

function double_integral_num(H, z, ξp, φ, ψ, κ, ν)
    quadgk(ξ -> Gu_num(z, ξ, H, ν, φ) * inner_b(ξ, ξp, H, κ, ψ),
           -H, 0.0; rtol=1e-10, atol=1e-12)[1]
end


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

# Example values
Hval  = 1.0
zval  = -0.6
ξpval = -0.5
φval  = 0.05 * sqrt(im)
# φval  = 0.05 
ψval  = 0.05
νval  = 1.0
κval  = 1.0

sym_val, num_val, abs_err, rel_err = compare_symbolic_vs_numeric(
    Guvb, Hval, zval, ξpval, φval, ψval, κval, νval
)

println("symbolic = ", sym_val)
println("numeric  = ", num_val)
@printf("abs err  = %.16e\n", abs_err)
@printf("rel err  = %.16e\n", rel_err)

# More comprehensive tests for several parameters:
Ek_vals = [1e-1, 1e-2, 1e-3, 1e-4]
z_vals = collect(range(-Hval*(0.999), -0.001; length=4))
ξp_vals = z_vals

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
        t_num = @elapsed num_val = double_integral_num(Hval, zval, ξpval, φval, ψval, κval, νval)
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

for Ekval in Ek_vals
    for zval in z_vals
        for ξpval in ξp_vals
            push!(results, benchmark_one(Ekval, Hval, zval, ξpval; νval=νval, κval=κval))
        end
    end
end

summarize_results(results)

println("\nPer-case summary:")
for r in results
    @printf("Ek=%-8.1e z=%-8.4f ξp=%-8.4f ok=%-5s  relerr=%-10.3e  t_sym=%-9.3e  t_num=%-9.3e\n",
            r.Ek, r.z, r.ξp, string(r.ok), r.rel_err, r.t_sym, r.t_num)
end