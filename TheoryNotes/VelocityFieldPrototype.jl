"""
Prototype implementation and comparison of the complex horizontal velocity
field in equations (B10) and (B23).

`velocity_b10` is a direct, nested-quadrature reference implementation.
`velocity_b23` uses the symbolic vertical kernels produced by
`ComputeQuadratures.ipynb` and is the production-oriented implementation.
"""
module VelocityFieldPrototype

using Printf
using QuadGK
using Statistics

export VelocityParameters,
       buoyancy_from_forcing,
       load_symbolic_kernels,
       velocity_b10,
       velocity_b23,
       velocity_b23_from_gradps,
       run_manufactured_test

const HERE = @__DIR__
const RTOL = 1e-9
const ATOL = 1e-11

include("VelocitySymbolicKernels.jl")

"""
    VelocityParameters(Ek, Γ, ν, κ, f0, β)

Parameters for the nondimensional, uniform-viscosity/diffusivity kernels.
`f(y) = f0 + β*y` must remain positive over the test domain.
"""
struct VelocityParameters{T<:Real}
    Ek::T
    Γ::T
    ν::T
    κ::T
    f0::T
    β::T
end

fcor(p::VelocityParameters, y) = p.f0 + p.β * y
φ(p::VelocityParameters, y) = sqrt(im * fcor(p, y) / (p.ν * p.Ek))
ψ(p::VelocityParameters) = sqrt(p.Γ / (p.κ * p.Ek))

function checked_state(p::VelocityParameters, H, y, z)
    H > 0 || throw(ArgumentError("H(x, y) must be positive; got $H"))
    fcor(p, y) > 0 || throw(ArgumentError("f(y) must be positive; got $(fcor(p, y))"))
    -H <= z <= 0 || throw(ArgumentError("z=$z lies outside [-H, 0]=[-$H, 0]"))
end

function split_quadgk(f, a, b, breaks=(); rtol=RTOL, atol=ATOL)
    points = sort!(unique!(vcat(a, [q for q in breaks if a < q < b], b)))
    value = zero(f((a + b) / 2))
    for i in 1:length(points)-1
        value += quadgk(f, points[i], points[i + 1]; rtol, atol)[1]
    end
    return value
end

"""
    Gu(z, ξ, H, ν, φ)

Velocity Green's function `G_u` from the uniform-viscosity solution. This
matches the expression checked in `SymbolicsTesting.jl`.
"""
function Gu(z, ξ, H, ν, φ)
    prefactor = exp(-φ * (z + ξ)) / (2 * ν * φ * (1 + exp(2 * φ * H)))
    if z <= ξ
        return prefactor * (
            exp(2 * φ * ξ) - exp(2 * φ * (z + H)) -
            exp(2 * φ * (z + ξ + H)) + 1
        )
    end
    return prefactor * (
        exp(2 * φ * z) - exp(2 * φ * (ξ + H)) -
        exp(2 * φ * (z + ξ + H)) + 1
    )
end

function Gb(z, ξ, H, κ, ψ)
    denominator = 2 * κ * ψ * (1 + exp(2 * ψ * H))
    prefactor = exp(ψ * (z + ξ)) / denominator
    if z <= ξ
        return prefactor * (1 - exp(-2 * ψ * ξ)) *
               (exp(2 * ψ * H) + exp(-2 * ψ * z))
    end
    return prefactor * (1 - exp(-2 * ψ * z)) *
           (exp(2 * ψ * H) + exp(-2 * ψ * ξ))
end

function dGb_dξ_at_surface(z, H, κ, ψ)
    return exp(ψ * z) * (1 + exp(-2 * ψ * (z + H))) /
           (κ * (1 + exp(-2 * ψ * H)))
end

"""
    buoyancy_from_forcing(p, H, B, bs, x, y, z)

Evaluate the buoyancy solution driven by the interior restoring field `B` and
surface buoyancy `bs` for one water-column point.
"""
function buoyancy_from_forcing(p, Hfun, B, bs, x, y, z)
    H = Hfun(x, y)
    checked_state(p, H, y, z)
    ψvalue = ψ(p)
    interior = split_quadgk(
        ξp -> Gb(z, ξp, H, p.κ, ψvalue) * B(x, y, ξp),
        -H, 0.0, (z,),
    )
    return -p.Γ / p.Ek * interior +
           dGb_dξ_at_surface(z, H, p.κ, ψvalue) * bs(x, y)
end

function Gu_integral(z, H, ν, φ)
    return (
        -1 - exp(2 * φ * H) + exp(φ * (z + H)) + exp(φ * (H - z))
    ) / (ν * φ^2 * (1 + exp(2 * φ * H)))
end

function inner_Gb(z, ξ, H, κ, ψ)
    return split_quadgk(zp -> Gb(zp, ξ, H, κ, ψ), z, 0.0, (ξ,))
end

"""
    finite_complex_gradient(field, x, y; step=1e-5)

Evaluate `(d/dx + i*d/dy)` of a scalar field with centered differences.
The manufactured test state has `H = H(y)`, matching the derivation of
the `G_u,y` terms in (B23).
"""
function finite_complex_gradient(field, x, y; step=1e-5)
    dx = (field(x + step, y) - field(x - step, y)) / (2 * step)
    dy = (field(x, y + step) - field(x, y - step)) / (2 * step)
    return dx + im * dy
end

function Pi_direct(p, Hfun, b, bs, x, y, ξ)
    H = Hfun(x, y)
    ψvalue = ψ(p)
    bsolution(zp) = -p.Γ / p.Ek *
        split_quadgk(ξp -> Gb(zp, ξp, H, p.κ, ψvalue) * b(x, y, ξp),
                      -H, 0.0, (zp,)) +
        dGb_dξ_at_surface(zp, H, p.κ, ψvalue) * bs(x, y)
    return split_quadgk(bsolution, ξ, 0.0, ())
end

"""
    velocity_b10(p, H, ps, τ, B, bs, x, y, z; gradient_step=1e-5)

Direct reference evaluation of (B10). `ps`, `τ`, and `bs` are horizontal
fields; `B(x, y, z)` is the interior buoyancy-restoring field. Its nested
quadratures intentionally make it unsuitable for production use.
"""
function velocity_b10(p, Hfun, ps, τ, B, bs, x, y, z; gradient_step=1e-5)
    H = Hfun(x, y)
    checked_state(p, H, y, z)
    φvalue = φ(p, y)

    pressure = Gu_integral(z, H, p.ν, φvalue) *
               finite_complex_gradient(ps, x, y; step=gradient_step)
    stress = -p.Ek * Gu(z, 0.0, H, p.ν, φvalue) * τ(x, y)
    baroclinic(ξ) = Gu(z, ξ, H, p.ν, φvalue) *
        finite_complex_gradient((xx, yy) -> Pi_direct(p, Hfun, B, bs, xx, yy, ξ),
                                x, y; step=gradient_step)
    return (pressure + stress +
            split_quadgk(baroclinic, -H, 0.0, (z,))) / p.Ek
end

struct SymbolicKernels{F1,F2,F3,F4,F5,F6}
    Guvb::F1
    Guvb_xi::F2
    Guv_y_b::F3
    Guv_y_b_xi::F4
    inner_Gb::F5
    inner_Gb_xi::F6
end

"""
    load_symbolic_kernels()

Load the analytical kernels exported by `ComputeQuadratures.ipynb`. Run that
notebook first if the generated kernel source needs to be regenerated.
"""
function load_symbolic_kernels()
    return SymbolicKernels(
        (H, z, ξp, φ, ψ, κ, ν) -> generated_guvb(H, ψ, ξp, φ, z, κ, ν),
        (H, z, φ, ψ, κ, ν) -> generated_guvb_xi(H, z, φ, ψ, κ, ν),
        (H, z, ξp, φ, ψ, κ, ν, β, f, H_y) ->
            generated_guv_y_b(H, z, ξp, φ, ψ, κ, ν, β, f, H_y),
        (H, z, φ, ψ, κ, ν, β, f, H_y) ->
            generated_guv_y_b_xi(H, z, φ, ψ, κ, ν, β, f, H_y),
        (ξ, ξp, H, ψ, κ) -> generated_inner_gb(H, ξ, ξp, ψ, κ),
        (ξ, H, ψ, κ) -> generated_inner_gb_xi(H, ξ, ψ, κ),
    )
end

function integrated_pressure_b23(kernels, p, Hfun, B, bs, x, y, z)
    H = Hfun(x, y)
    φvalue, ψvalue = φ(p, y), ψ(p)
    return -p.Γ / p.Ek * split_quadgk(
        ξp -> kernels.Guvb(H, z, ξp, φvalue, ψvalue, p.κ, p.ν) * B(x, y, ξp),
        -H, 0.0, (z,),
    ) + kernels.Guvb_xi(H, z, φvalue, ψvalue, p.κ, p.ν) * bs(x, y)
end

function bottom_pressure(kernels, p, Hfun, B, bs, x, y)
    H = Hfun(x, y)
    ψvalue = ψ(p)
    return -p.Γ / p.Ek * split_quadgk(
        ξp -> kernels.inner_Gb(-H, ξp, H, ψvalue, p.κ) * B(x, y, ξp),
        -H, 0.0, (),
    ) + kernels.inner_Gb_xi(-H, H, ψvalue, p.κ) * bs(x, y)
end

"""
    velocity_b23(kernels, p, H, ps, τ, B, bs, H_x, H_y, x, y, z;
                 gradient_step=1e-5)

Production-oriented evaluation of (B23): three single vertical quadratures
plus closed-form symbolic kernels. `H_x` and `H_y` are analytic bathymetric
slopes. The x-derivative kernel is obtained from the H_y component of the
generated G_u,y kernel by setting beta=0 and H_y=1.
"""
function velocity_b23(kernels, p, Hfun, ps, τ, B, bs, H_x, H_y, x, y, z;
                      gradient_step=1e-5)
    ps_gradient(xx, yy) = finite_complex_gradient(ps, xx, yy; step=gradient_step)
    return velocity_b23_from_gradps(kernels, p, Hfun, ps_gradient, τ, B, bs,
                                    H_x, H_y, x, y, z; gradient_step)
end

"""
    velocity_b23_from_gradps(kernels, p, H, gradps, τ, B, bs, H_x, H_y, x, y, z;
                             gradient_step=1e-5)

Evaluate (B23) when `gradps(x,y)` already provides `(d/dx + i*d/dy) p_s`.
Use this form with a finite-element pressure solve to avoid finite
differences of the pressure interpolant.
"""
function velocity_b23_from_gradps(kernels, p, Hfun, gradps, τ, B, bs, H_x, H_y,
                                  x, y, z; gradient_step=1e-5)
    H = Hfun(x, y)
    checked_state(p, H, y, z)
    φvalue, ψvalue = φ(p, y), ψ(p)

    pressure = Gu_integral(z, H, p.ν, φvalue) * gradps(x, y)
    stress = -p.Ek * Gu(z, 0.0, H, p.ν, φvalue) * τ(x, y)
    integrated = finite_complex_gradient(
        (xx, yy) -> integrated_pressure_b23(kernels, p, Hfun, B, bs, xx, yy, z),
        x, y; step=gradient_step,
    )
    bottom = finite_complex_gradient(Hfun, x, y; step=gradient_step) *
             Gu(z, -H, H, p.ν, φvalue) *
             bottom_pressure(kernels, p, Hfun, B, bs, x, y)
    kernel_x = H_x(x, y) * (
        -p.Γ / p.Ek * split_quadgk(
            ξp -> kernels.Guv_y_b(H, z, ξp, φvalue, ψvalue, p.κ, p.ν,
                                   0.0, 1.0, 1.0) * B(x, y, ξp),
            -H, 0.0, (z,),
        ) + kernels.Guv_y_b_xi(H, z, φvalue, ψvalue, p.κ, p.ν,
                                0.0, 1.0, 1.0) * bs(x, y)
    )
    kernel_y = -p.Γ / p.Ek * split_quadgk(
        ξp -> kernels.Guv_y_b(H, z, ξp, φvalue, ψvalue, p.κ, p.ν,
                               p.β, fcor(p, y), H_y(x, y)) * B(x, y, ξp),
        -H, 0.0, (z,),
    ) + kernels.Guv_y_b_xi(H, z, φvalue, ψvalue, p.κ, p.ν,
                            p.β, fcor(p, y), H_y(x, y)) * bs(x, y)
    return (pressure + stress + integrated - bottom - kernel_x - im * kernel_y) / p.Ek
end

function benchmark(f; samples=5)
    f() # compilation and cache warm-up
    times = [@elapsed f() for _ in 1:samples]
    return median(times)
end

"""
    run_manufactured_test(; samples=5, rtol=2e-5)

Compare (B10) and (B23) at one smooth manufactured state and print timing
and agreement. It returns the values and timing information for use from a
notebook or a production-driver test.
"""
function run_manufactured_test(; samples=5, rtol=2e-5)
    p = VelocityParameters(1e-2, 1.0, 1.0, 1.0, 1.0, 0.15)
    H(x, y) = 1.0 + 0.04 * x + 0.08 * y
    H_x(x, y) = 0.04
    H_y(x, y) = 0.08
    ps(x, y) = sin(0.7 * x) * cos(0.4 * y)
    τ(x, y) = 0.03 * (1 + 0.2im) * cos(0.3 * x - 0.5 * y)
    B(x, y, z) = exp(z) * (1 + 0.15 * sin(x) * cos(y))
    bs(x, y) = 0.04 * cos(0.6 * x + 0.2 * y)
    x, y, z = 0.25, 0.3, -0.45

    kernels = load_symbolic_kernels()
    b10() = velocity_b10(p, H, ps, τ, B, bs, x, y, z)
    b23() = velocity_b23(kernels, p, H, ps, τ, B, bs, H_x, H_y, x, y, z)
    u_b10, u_b23 = b10(), b23()
    relative_error = abs(u_b10 - u_b23) / max(abs(u_b10), eps())
    relative_error <= rtol || throw(ErrorException(
        "(B10) and (B23) differ by $(relative_error), exceeding rtol=$rtol. " *
        "Check the manufactured state and kernel export."
    ))

    b10_time = benchmark(b10; samples)
    b23_time = benchmark(b23; samples)
    @printf("B10 velocity: % .10e %+.10ei\n", real(u_b10), imag(u_b10))
    @printf("B23 velocity: % .10e %+.10ei\n", real(u_b23), imag(u_b23))
    @printf("relative error: %.3e\n", relative_error)
    @printf("median time B10: %.3e s; B23: %.3e s; speedup: %.2fx\n",
            b10_time, b23_time, b10_time / b23_time)
    return (; u_b10, u_b23, relative_error, b10_time, b23_time,
            speedup=b10_time / b23_time)
end

end # module
