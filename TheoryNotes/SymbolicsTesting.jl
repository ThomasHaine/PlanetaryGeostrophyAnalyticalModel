using Symbolics
using JLD2

@variables x y z ξ 
@variables H(..)
@variables HH
@variables Ek κ ν ψ φ

@load "symexpr.jld2" symbolics_expr

tmp = Symbolics.substitute(symbolics_expr, Dict(ν => 1, κ => 1, H(x,y) => HH))
println("Substituted expression:")
display(tmp)

vars = Symbolics.get_variables(symbolics_expr)
println("Variables in the expression:")
display(vars)

Guvb = build_function(tmp, vars; expression=Val{false}, eval_expression=Val{true})

println("Generated function:")
display(Guvb)