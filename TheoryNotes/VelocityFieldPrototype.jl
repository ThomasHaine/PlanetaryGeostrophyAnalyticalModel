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
       buoyancy_bottom_stress_b23,
       buoyancy_transport_b23,
       buoyancy_transport_sensitivity_coefficients,
       buoyancy_column_integral,
       load_symbolic_kernels,
       velocity_b10,
       velocity_b23,
       velocity_b23_column_from_gradps,
       velocity_b23_from_gradps,
       run_manufactured_test

const HERE = @__DIR__
const RTOL = 1e-9
const ATOL = 1e-11

include("VelocitySymbolicKernels.jl")

"""
    VelocityParameters(Ek, Γ, ν, κ, f0, β)

Parameters for the nondimensional, uniform-viscosity/diffusivity kernels.
The `f0` and `β` fields retain the beta-plane parameters used by the B10
Reference implementation. B23 accepts an arbitrary sign-definite, nonzero
`f(x,y)` and its
horizontal derivatives directly.
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
φ_from_f(p::VelocityParameters, f) = sqrt(im * f / (p.ν * p.Ek))
ψ(p::VelocityParameters) = sqrt(p.Γ / (p.κ * p.Ek))

function checked_depth(H, z)
    H > 0 || throw(ArgumentError("H(x, y) must be positive; got $H"))
    -H <= z <= 0 || throw(ArgumentError("z=$z lies outside [-H, 0]=[-$H, 0]"))
end

function checked_state(p::VelocityParameters, H, y, z)
    checked_depth(H, z)
    fcor(p, y) != 0 || throw(ArgumentError("f(y) must be nonzero; got $(fcor(p, y))"))
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
    checked_depth(H, z)
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
    velocity_b10(p, H, f, ps, τ, B, bs, x, y, z; gradient_step=1e-5)

Direct reference evaluation of (B10). `ps`, `τ`, and `bs` are horizontal
fields; `B(x, y, z)` is the interior buoyancy-restoring field. Its nested
quadratures intentionally make it unsuitable for production use.
"""
function velocity_b10(p, Hfun, ffun, ps, τ, B, bs, x, y, z; gradient_step=1e-5)
    H = Hfun(x, y)
    checked_depth(H, z)
    fvalue = ffun(x, y)
    fvalue != 0 || throw(ArgumentError("f(x, y) must be nonzero; got $fvalue"))
    φvalue = φ_from_f(p, fvalue)

    pressure = Gu_integral(z, H, p.ν, φvalue) *
               finite_complex_gradient(ps, x, y; step=gradient_step)
    stress = -p.Ek * Gu(z, 0.0, H, p.ν, φvalue) * τ(x, y)
    baroclinic(ξ) = Gu(z, ξ, H, p.ν, φvalue) *
        finite_complex_gradient((xx, yy) -> Pi_direct(p, Hfun, B, bs, xx, yy, ξ),
                                x, y; step=gradient_step)
    return (pressure + stress -
            split_quadgk(baroclinic, -H, 0.0, (z,))) / p.Ek
end

function velocity_b10(p, Hfun, ps, τ, B, bs, x, y, z; gradient_step=1e-5)
    ffun(xx, yy) = fcor(p, yy)
    return velocity_b10(p, Hfun, ffun, ps, τ, B, bs, x, y, z; gradient_step)
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

function integrated_pressure_b23(kernels, p, Hfun, ffun, B, bs, x, y, z)
    H = Hfun(x, y)
    φvalue, ψvalue = φ_from_f(p, ffun(x, y)), ψ(p)
    return -p.Γ / p.Ek * split_quadgk(
        ξp -> kernels.Guvb(H, z, ξp, φvalue, ψvalue, p.κ, p.ν) * B(x, y, ξp),
        -H, 0.0, (z,),
    ) + kernels.Guvb_xi(H, z, φvalue, ψvalue, p.κ, p.ν) * bs(x, y)
end

"""
    buoyancy_column_integral(kernels, p, H, B, bs, x, y)

Evaluate `Π(-H) = ∫_{-H}^0 b dz`. This is the buoyancy-column contribution
subtracted from surface pressure to obtain total bottom pressure.
"""
function buoyancy_column_integral(kernels, p, Hfun, B, bs, x, y)
    H = Hfun(x, y)
    ψvalue = ψ(p)
    return -p.Γ / p.Ek * split_quadgk(
        ξp -> kernels.inner_Gb(-H, ξp, H, ψvalue, p.κ) * B(x, y, ξp),
        -H, 0.0, (),
    ) + kernels.inner_Gb_xi(-H, H, ψvalue, p.κ) * bs(x, y)
end

"""
    buoyancy_transport_b23(kernels, p, H, f, f_x, f_y, B, bs, H_x, H_y, x, y)

Evaluate the buoyancy-driven depth-integrated transport in equation (69).
This is a general numerical B23 implementation that is suitable for checking
new forcing fields before specialized vertically integrated kernels are added.
"""
function buoyancy_transport_b23(kernels, p, Hfun, ffun, f_x, f_y, B, bs, H_x, H_y, x, y)
    H = Hfun(x, y)
    checked_depth(H, -H)
    zero_gradient(_, _) = 0.0 + 0.0im
    zero_stress(_, _) = 0.0 + 0.0im
    return split_quadgk(
        z -> velocity_b23_from_gradps(
            kernels, p, Hfun, ffun, f_x, f_y, zero_gradient, zero_stress,
            B, bs, H_x, H_y, x, y, z,
        ),
        -H, 0.0,
    )
end

"""
    buoyancy_transport_sensitivity_coefficients(
        kernels, p, H, f, f_x, f_y, H_x, H_y, x, y, ξ,
    )

Return the local coefficients that map interior and surface buoyancy values
and horizontal gradients to the complex depth-integrated transport forcing.
For an interior level `ξ`,

`δB_transport = value * δB + gradient * (δB_x + im * δB_y)`.

The `surface` entry has the analogous coefficients for `δb_s`. For sloping
topography, integrating the interior coefficients over `ξ` also requires the
bottom-boundary contribution
`interior.gradient(ξ=-H) * (H_x + im * H_y) * δB(x,y,-H)`.
"""
function buoyancy_transport_sensitivity_coefficients(
    kernels,
    p,
    Hfun,
    ffun,
    f_x,
    f_y,
    H_x,
    H_y,
    x,
    y,
    ξ,
    ;
    gradient_step=1e-5,
)
    H = Hfun(x, y)
    checked_depth(H, ξ)
    fvalue = ffun(x, y)
    fvalue != 0 || throw(ArgumentError("f(x, y) must be nonzero; got $fvalue"))
    φvalue, ψvalue = φ_from_f(p, fvalue), ψ(p)
    Hx, Hy = H_x(x, y), H_y(x, y)
    fx, fy = f_x(x, y), f_y(x, y)

    integrated_Guvb = split_quadgk(
        z -> kernels.Guvb(
            H, z, ξ, φvalue, ψvalue, p.κ, p.ν,
        ),
        -H, 0.0, (ξ,),
    )
    integrated_Guvb_xi = split_quadgk(
        z -> kernels.Guvb_xi(
            H, z, φvalue, ψvalue, p.κ, p.ν,
        ),
        -H, 0.0, (),
    )
    function interior_gradient_coefficient(xx, yy)
        local_H = Hfun(xx, yy)
        local_φ = φ_from_f(p, ffun(xx, yy))
        return p.Γ / p.Ek^2 * split_quadgk(
            z -> kernels.Guvb(
                local_H, z, ξ, local_φ, ψvalue, p.κ, p.ν,
            ),
            -local_H, 0.0, (ξ,),
        )
    end
    function surface_gradient_coefficient(xx, yy)
        local_H = Hfun(xx, yy)
        local_φ = φ_from_f(p, ffun(xx, yy))
        return -split_quadgk(
            z -> kernels.Guvb_xi(
                local_H, z, local_φ, ψvalue, p.κ, p.ν,
            ),
            -local_H, 0.0, (),
        ) / p.Ek
    end
    interior_coefficient_gradient = finite_complex_gradient(
        interior_gradient_coefficient, x, y; step=gradient_step,
    )
    surface_coefficient_gradient = finite_complex_gradient(
        surface_gradient_coefficient, x, y; step=gradient_step,
    )
    integrated_bottom_velocity = split_quadgk(
        z -> Gu(z, -H, H, p.ν, φvalue),
        -H, 0.0, (),
    )

    function integrated_directional_kernel(
        H_derivative,
        f_derivative,
    )
        return split_quadgk(
            z -> kernels.Guv_y_b(
                H, z, ξ, φvalue, ψvalue, p.κ, p.ν,
                f_derivative, fvalue, H_derivative,
            ),
            -H, 0.0, (ξ,),
        )
    end
    function integrated_directional_surface_kernel(
        H_derivative,
        f_derivative,
    )
        return split_quadgk(
            z -> kernels.Guv_y_b_xi(
                H, z, φvalue, ψvalue, p.κ, p.ν,
                f_derivative, fvalue, H_derivative,
            ),
            -H, 0.0, (),
        )
    end

    column = -p.Γ / p.Ek *
        kernels.inner_Gb(-H, ξ, H, ψvalue, p.κ)
    surface_column =
        kernels.inner_Gb_xi(-H, H, ψvalue, p.κ)
    bottom_factor =
        (Hx + im * Hy) * integrated_bottom_velocity / p.Ek
    interior = (
        value=(
            interior_coefficient_gradient +
            bottom_factor * column -
            p.Γ / p.Ek^2 * (
                integrated_directional_kernel(Hx, fx) +
                im * integrated_directional_kernel(Hy, fy)
            )
        ),
        gradient=p.Γ / p.Ek^2 * integrated_Guvb,
    )
    surface = (
        value=(
            surface_coefficient_gradient +
            bottom_factor * surface_column +
            (
                integrated_directional_surface_kernel(Hx, fx) +
                im * integrated_directional_surface_kernel(Hy, fy)
            ) / p.Ek
        ),
        gradient=-integrated_Guvb_xi / p.Ek,
    )
    return (; interior, surface)
end

"""
    buoyancy_bottom_stress_b23(kernels, p, H, f, f_x, f_y, B, bs, H_x, H_y, x, y;
                                vertical_step=1e-4)

Evaluate the buoyancy-driven bottom stress in equation (65), using the
one-sided second-order approximation `-ν ∂u/∂z` at `z = -H`. It intentionally
uses B23 velocity evaluations so variable bathymetry and general `f(x,y)` are
treated consistently with the production velocity calculation.
"""
function buoyancy_bottom_stress_b23(
    kernels, p, Hfun, ffun, f_x, f_y, B, bs, H_x, H_y, x, y;
    vertical_step=1e-4,
)
    H = Hfun(x, y)
    checked_depth(H, -H)
    h = min(vertical_step, H / 4)
    h > 0 || throw(ArgumentError("vertical_step must be positive; got $vertical_step"))
    zero_gradient(_, _) = 0.0 + 0.0im
    zero_stress(_, _) = 0.0 + 0.0im
    velocity(z) = velocity_b23_from_gradps(
        kernels, p, Hfun, ffun, f_x, f_y, zero_gradient, zero_stress,
        B, bs, H_x, H_y, x, y, z,
    )
    return -p.ν * (-3 * velocity(-H) + 4 * velocity(-H + h) -
                   velocity(-H + 2h)) / (2h)
end

"""
    velocity_b23(kernels, p, H, f, ps, τ, B, bs, H_x, H_y, x, y, z;
                 gradient_step=1e-5)

Production-oriented evaluation of (B23): three single vertical quadratures
plus closed-form symbolic kernels. `H_x` and `H_y` are analytic bathymetric
slopes. The x-derivative kernel is obtained from the H_y component of the
generated G_u,y kernel by setting beta=0 and H_y=1.
"""
function velocity_b23(kernels, p, Hfun, ffun, ps, τ, B, bs, H_x, H_y, x, y, z;
                      gradient_step=1e-5)
    ps_gradient(xx, yy) = finite_complex_gradient(ps, xx, yy; step=gradient_step)
    f_x(xx, yy) = (ffun(xx + gradient_step, yy) - ffun(xx - gradient_step, yy)) /
                  (2 * gradient_step)
    f_y(xx, yy) = (ffun(xx, yy + gradient_step) - ffun(xx, yy - gradient_step)) /
                  (2 * gradient_step)
    return velocity_b23_from_gradps(kernels, p, Hfun, ffun, f_x, f_y, ps_gradient,
                                    τ, B, bs, H_x, H_y, x, y, z; gradient_step)
end

"""
    velocity_b23_from_gradps(kernels, p, H, f, f_x, f_y, gradps, τ, B, bs,
                             H_x, H_y, x, y, z; gradient_step=1e-5)

Evaluate (B23) when `gradps(x,y)` already provides `(d/dx + i*d/dy) p_s`.
Use this form with a finite-element pressure solve to avoid finite
differences of the pressure interpolant. `f_x` and `f_y` are derivatives of
the generalized Coriolis parameter.
"""
function velocity_b23_from_gradps(kernels, p, Hfun, ffun, f_x, f_y, gradps, τ, B,
                                  bs, H_x, H_y, x, y, z; gradient_step=1e-5)
    velocity = velocity_b23_column_from_gradps(
        kernels, p, Hfun, ffun, f_x, f_y, gradps, τ, B, bs,
        H_x, H_y, x, y; gradient_step,
    )
    return velocity(z)
end

"""
    velocity_b23_column_from_gradps(
        kernels, p, H, f, f_x, f_y, gradps, τ, B, bs, H_x, H_y, x, y;
        gradient_step=1e-5,
    )

Prepare an analytical B23 velocity evaluator for one water column. Quantities
that depend only on `(x, y)`, including the pressure gradient and vertically
integrated buoyancy, are evaluated once and reused for every requested depth.
"""
function velocity_b23_column_from_gradps(
    kernels, p, Hfun, ffun, f_x, f_y, gradps, τ, B,
    bs, H_x, H_y, x, y; gradient_step=1e-5,
)
    H = Hfun(x, y)
    fvalue = ffun(x, y)
    checked_depth(H, 0.0)
    fvalue != 0 || throw(ArgumentError("f(x, y) must be nonzero; got $fvalue"))
    φvalue, ψvalue = φ_from_f(p, fvalue), ψ(p)
    pressure_gradient = gradps(x, y)
    surface_stress = τ(x, y)
    Hx, Hy = H_x(x, y), H_y(x, y)
    fx, fy = f_x(x, y), f_y(x, y)
    column_buoyancy =
        buoyancy_column_integral(kernels, p, Hfun, B, bs, x, y)
    surface_buoyancy = bs(x, y)

    function velocity(z)
        checked_depth(H, z)
        pressure =
            Gu_integral(z, H, p.ν, φvalue) * pressure_gradient
        stress =
            -p.Ek * Gu(z, 0.0, H, p.ν, φvalue) * surface_stress
        integrated = finite_complex_gradient(
            (xx, yy) -> integrated_pressure_b23(
                kernels, p, Hfun, ffun, B, bs, xx, yy, z,
            ),
            x, y; step=gradient_step,
        )
        bottom = (Hx + im * Hy) * Gu(z, -H, H, p.ν, φvalue) *
            column_buoyancy

        function directional_kernel(H_derivative, f_derivative)
            -p.Γ / p.Ek * split_quadgk(
                ξp -> kernels.Guv_y_b(
                    H, z, ξp, φvalue, ψvalue, p.κ, p.ν,
                    f_derivative, fvalue, H_derivative,
                ) * B(x, y, ξp),
                -H, 0.0, (z,),
            ) + kernels.Guv_y_b_xi(
                H, z, φvalue, ψvalue, p.κ, p.ν,
                f_derivative, fvalue, H_derivative,
            ) * surface_buoyancy
        end

        kernel_x = directional_kernel(Hx, fx)
        kernel_y = directional_kernel(Hy, fy)
        return (
            pressure + stress - integrated + bottom +
            kernel_x + im * kernel_y
        ) / p.Ek
    end
    return velocity
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
    f(x, y) = fcor(p, y)
    ps(x, y) = sin(0.7 * x) * cos(0.4 * y)
    τ(x, y) = 0.03 * (1 + 0.2im) * cos(0.3 * x - 0.5 * y)
    B(x, y, z) = exp(z) * (1 + 0.15 * sin(x) * cos(y))
    bs(x, y) = 0.04 * cos(0.6 * x + 0.2 * y)
    x, y, z = 0.25, 0.3, -0.45

    kernels = load_symbolic_kernels()
    b10() = velocity_b10(p, H, f, ps, τ, B, bs, x, y, z)
    b23() = velocity_b23(kernels, p, H, f, ps, τ, B, bs, H_x, H_y, x, y, z)
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
