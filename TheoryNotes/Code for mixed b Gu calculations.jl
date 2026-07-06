# Construct the buoyancy related T(x) term (excludes the stress-driven part):
Txb_term_integrand = exp(-root_iϕ * (ξ - H(x,y))) * (1 + exp(2*root_iϕ * ξ)) / (1 + exp(2*root_iϕ * H(x,y)))
Txb_term_integrand = Txb_term_integrand * dGb_int_int_∂z.subs(z,ξ)
Txb_term = sympy.integrate(sympy.expand(Txb_term_integrand), (ξ, -H(x,y), 0))
Txb_term = sum(Txb_term.args[i] for i in 1:length(Txb_term.args)-1) + Txb_term.args[length(Txb_term.args)].args[1].args[1]    # Manually expand the integral result

Latex_Txb_term = α * Γ * (exp((ψ + root_iϕ) * H(x,y)) - exp((3*ψ + root_iϕ) * H(x,y)))/
(Ek * ψ^3 * (1 + exp(2 * ψ * H(x,y)))^2 * (1 + exp(2 * root_iϕ * H(x,y)) ))
Latex_Txb_term = Latex_Txb_term * (sympy.diff(H(x,y),x) + im*sympy.diff(H(x,y),y))
Latex_int_result = (
((1 - exp(- (ψ - root_iϕ)*H(x,y)))/( ψ - root_iϕ)) +
((1 - exp(- (ψ + root_iϕ)*H(x,y)))/( ψ + root_iϕ)) +
(2/root_iϕ) * (exp(-root_iϕ * H(x,y)) - exp(root_iϕ * H(x,y))) +
((1 - exp(  (ψ + root_iϕ)*H(x,y)))/(-ψ - root_iϕ)) + 
((1 - exp(  (ψ - root_iϕ)*H(x,y)))/(-ψ + root_iϕ)) 
)
# Check intermediate analytic Latex integral:
inter1 = exp(- (ψ - root_iϕ)*ξ) + exp(- (ψ + root_iϕ)*ξ) - 2*(exp(-root_iϕ*ξ) + exp(+root_iϕ*ξ)) + exp((ψ + root_iϕ)*ξ) + exp((ψ - root_iϕ)*ξ)
inter2 = sympy.integrate(inter1, (ξ,-H(x,y),0))
@assert inter2 - Latex_int_result == 0
Latex_Txb_term = Latex_Txb_term * Latex_int_result
# Substitute specific parameter values to check equality, otherwise SymPy takes a very long time...
tmp = (sympy.expand(Latex_Txb_term) - sympy.expand(Txb_term)).subs(sample_params).doit()  # For easier comparison
tmp = sympy.simplify(tmp)
@assert tmp == 0
println("1.4 LaTeX T(x) buoyancy term with linear buoyancy relaxation profile matches.")
display(Latex_Txb_term)

# Construct the buoyancy related B(x) term (excludes the stress-driven part):
Bxb_term_integrand = exp(-root_iϕ * ξ) * (exp(root_iϕ * ξ) - exp(root_iϕ * H(x,y))) * (exp(root_iϕ * (ξ + H(x,y))) - 1)
Bxb_term_integrand = Bxb_term_integrand * dGb_int_int_∂z.subs(z,ξ)
Bxb_term = sympy.integrate(sympy.expand(Bxb_term_integrand), (ξ, -H(x,y), 0))

Latex_Bxb_term = α * Γ * (exp(ψ * H(x,y)) - exp(3 * ψ * H(x,y)))/(Ek * ψ^3 * (1 + exp(2 * ψ * H(x,y)))^2)
Latex_Bxb_term = Latex_Bxb_term * (sympy.diff(H(x,y),x) + im*sympy.diff(H(x,y),y))
Latex_int_result = exp(root_iϕ * H(x,y)) * (
((1 - exp(- (ψ - root_iϕ)*H(x,y)))/( ψ - root_iϕ)) +
((1 - exp(- (ψ + root_iϕ)*H(x,y)))/( ψ + root_iϕ)) +
(2/root_iϕ) * (exp(-root_iϕ * H(x,y)) - exp(root_iϕ * H(x,y))) +
((1 - exp(  (ψ + root_iϕ)*H(x,y)))/(-ψ - root_iϕ)) + 
((1 - exp(  (ψ - root_iϕ)*H(x,y)))/(-ψ + root_iϕ)) +
((exp((-ψ + root_iϕ) * H(x,y)) - exp((ψ + root_iϕ) * H(x,y)))/ψ)
)
Latex_int_result = Latex_int_result + (1/ψ)*(exp(-ψ * H(x,y)) - exp(ψ * H(x,y))) + 2*H(x,y)*(1 + exp(2*root_iϕ * H(x,y)))
# Check intermediate analytic Latex integral:
inter1 = (1 - exp(root_iϕ * (H(x,y) - ξ))) * (exp(root_iϕ * (H(x,y) + ξ)) - 1) * (exp(ψ * ξ) - 2 + exp(- ψ * ξ))
inter2 = sympy.integrate(sympy.expand(inter1), (ξ,-H(x,y),0))
@assert inter2 - Latex_int_result == 0
Latex_Bxb_term = Latex_Bxb_term * Latex_int_result
# Substitute specific parameter values to check equality, otherwise SymPy takes a very long time...
tmp = (sympy.expand(Latex_Bxb_term) - Bxb_term).subs(sample_params).doit()  # For easier comparison
tmp = sympy.simplify(tmp)
@assert tmp == 0
println("1.5 LaTeX B(x) buoyancy term with linear buoyancy relaxation profile matches.")
display(Latex_Bxb_term)