using BSplineKit
using LinearAlgebra

# ----------------------------------------------------------------------------
# 1D cubic spline fit
# ----------------------------------------------------------------------------

function fit_cubic_spline_1d(x, f)
    x = collect(x)
    @assert length(x) == length(f)
    @assert issorted(x)
    @assert all(isfinite, x)
    @assert all(isfinite, f)

    return interpolate(x, f, BSplineOrder(4))
end

# ----------------------------------------------------------------------------
# Cached 1D basis matrices built from the working interpolation path
# ----------------------------------------------------------------------------

"""
    bspline_collocation_matrix(xgrid)

Build A[i,m] = B_m(xgrid[i]) by interpolating Kronecker-delta data.
This is expensive once, but safe and compatible.
"""
function bspline_collocation_matrix(xgrid)
    xgrid = collect(xgrid)
    n = length(xgrid)
    @assert n ≥ 4

    A = Matrix{Float64}(undef, n, n)
    e = zeros(Float64, n)

    @inbounds for m in 1:n
        fill!(e, 0.0)
        e[m] = 1.0
        spl = fit_cubic_spline_1d(xgrid, e)
        for i in 1:n
            A[i, m] = spl(xgrid[i])
        end
    end

    return A
end

"""
    bspline_deriv_matrix(xgrid)

Build D[i,m] = B'_m(xgrid[i]) by interpolating Kronecker-delta data
and differentiating the spline.
"""
function bspline_deriv_matrix(xgrid)
    xgrid = collect(xgrid)
    n = length(xgrid)
    @assert n ≥ 4

    D = Matrix{Float64}(undef, n, n)
    e = zeros(Float64, n)

    @inbounds for m in 1:n
        fill!(e, 0.0)
        e[m] = 1.0
        spl = fit_cubic_spline_1d(xgrid, e)
        dspl = BSplineKit.Derivative(1) * spl
        for i in 1:n
            D[i, m] = dspl(xgrid[i])
        end
    end

    return D
end

# ----------------------------------------------------------------------------
# Cached tensor basis
# ----------------------------------------------------------------------------

struct TensorSplineBasis
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    Ax::Matrix{Float64}
    Ay::Matrix{Float64}
    Dx::Matrix{Float64}
    Dy::Matrix{Float64}
end

function build_tensor_spline_basis(xgrid, ygrid)
    xgrid = collect(xgrid)
    ygrid = collect(ygrid)

    @assert issorted(xgrid) && issorted(ygrid)
    @assert all(isfinite, xgrid) && all(isfinite, ygrid)
    @assert length(xgrid) ≥ 4 && length(ygrid) ≥ 4

    Ax = bspline_collocation_matrix(xgrid)
    Ay = bspline_collocation_matrix(ygrid)
    Dx = bspline_deriv_matrix(xgrid)
    Dy = bspline_deriv_matrix(ygrid)

    return TensorSplineBasis(xgrid, ygrid, Ax, Ay, Dx, Dy)
end

# ----------------------------------------------------------------------------
# Fit a tensor-product spline surface
# ----------------------------------------------------------------------------

"""
    fit_tensor_spline_2d(basis, F)

Fit tensor-product coefficients C from grid data F using:
    Ax * C * Ay' = F
"""
function fit_tensor_spline_2d(basis::TensorSplineBasis, F)
    @assert size(F) == (length(basis.xgrid), length(basis.ygrid))
    @assert all(isfinite, F)

    C = basis.Ax \ Matrix(F) / transpose(basis.Ay)
    return C
end

# ----------------------------------------------------------------------------
# Grid evaluation
# ----------------------------------------------------------------------------

eval_tensor_spline_grid(basis::TensorSplineBasis, C::AbstractMatrix) =
    basis.Ax * C * transpose(basis.Ay)

eval_tensor_spline_dx_grid(basis::TensorSplineBasis, C::AbstractMatrix) =
    basis.Dx * C * transpose(basis.Ay)

eval_tensor_spline_dy_grid(basis::TensorSplineBasis, C::AbstractMatrix) =
    basis.Ax * C * transpose(basis.Dy)

function eval_tensor_spline_complex_grad_grid(basis::TensorSplineBasis, C::AbstractMatrix)
    Fx = eval_tensor_spline_dx_grid(basis, C)
    Fy = eval_tensor_spline_dy_grid(basis, C)
    return Fx .+ im .* Fy
end

# ----------------------------------------------------------------------------
# Stack of ξ-slices
# ----------------------------------------------------------------------------

"""
    build_horizontal_tensor_splines(xgrid, ygrid, J)

Fit one coefficient matrix per ξ-slice.

Returns:
- coeffs::Vector{Matrix{Float64}}
- basis::TensorSplineBasis
"""
function build_horizontal_tensor_splines(xgrid, ygrid, J)
    nx, ny, nξ = size(J)
    basis = build_tensor_spline_basis(xgrid, ygrid)

    @assert nx == length(basis.xgrid)
    @assert ny == length(basis.ygrid)
    @assert all(isfinite, J)

    coeffs = Vector{Matrix{Float64}}(undef, nξ)

    @inbounds for k in 1:nξ
        coeffs[k] = fit_tensor_spline_2d(basis, view(J, :, :, k))
    end

    return coeffs, basis
end

"""
    build_complex_horizontal_gradient_slices(xgrid, ygrid, J)

For each ξ-slice, compute (∂x + i∂y)J on the tensor grid.

Returns:
- Q::Array{ComplexF64,3}
- coeffs::Vector{Matrix{Float64}}
- basis::TensorSplineBasis
"""
function build_complex_horizontal_gradient_slices(xgrid, ygrid, J)
    nx, ny, nξ = size(J)
    basis = build_tensor_spline_basis(xgrid, ygrid)

    @assert nx == length(basis.xgrid)
    @assert ny == length(basis.ygrid)
    @assert all(isfinite, J)

    Q = Array{ComplexF64}(undef, nx, ny, nξ)
    coeffs = Vector{Matrix{Float64}}(undef, nξ)

    @inbounds for k in 1:nξ
        F = view(J, :, :, k)
        C = fit_tensor_spline_2d(basis, F)
        coeffs[k] = C

        Fx = basis.Dx * C * transpose(basis.Ay)
        Fy = basis.Ax * C * transpose(basis.Dy)
        Q[:, :, k] = Fx .+ im .* Fy
    end

    return Q, coeffs, basis
end

# ----------------------------------------------------------------------------
# Vertical integration from BSplineKit z-splines
# ----------------------------------------------------------------------------

"""
    vertical_integrated_field_from_splines(Bzsplines, ξgrid)

Compute J[i,j,k] = ∫_{ξ_k}^0 B(i,j,z') dz' using spline integrals.
"""
function vertical_integrated_field_from_splines(Bzsplines, ξgrid)
    nx, ny = size(Bzsplines)
    nξ = length(ξgrid)

    J = Array{Float64}(undef, nx, ny, nξ)

    @inbounds for i in 1:nx, j in 1:ny
        spl = Bzsplines[i, j]
        A = BSplineKit.integral(spl)
        A0 = A(0.0)
        for k in 1:nξ
            ξ = ξgrid[k]
            J[i, j, k] = A0 - A(ξ)
        end
    end

    return J
end

# ----------------------------------------------------------------------------
# Extract coefficients of a cubic spline in shifted form
# ----------------------------------------------------------------------------

"""
    spline_interval_power_coeffs(spl)

Return:
- knots::Vector{Float64}
- coeffs::Vector{NTuple{4,Float64}}

where coeffs[k] are the power-basis coefficients on [knots[k], knots[k+1]]:

    B(ξ) = c0 + c1*ξ + c2*ξ^2 + c3*ξ^3
"""
function spline_interval_power_coeffs(spl)
    s = spl.spline  # unwrap SplineInterpolation -> Spline

    knots = s.basis.t
    nint = length(knots) - 1

    coeffs = Vector{NTuple{4,Float64}}(undef, nint)

    d1 = BSplineKit.Derivative(1) * s
    d2 = BSplineKit.Derivative(2) * s
    d3 = BSplineKit.Derivative(3) * s

    for k in 1:nint
        a = knots[k]
        b = knots[k+1]

        # midpoint of interval
        x0 = (a + b) / 2

        # shifted Taylor coefficients around x0
        a0 = s(x0)
        a1 = d1(x0)
        a2 = d2(x0) / 2
        a3 = d3(x0) / 6

        # convert shifted cubic to power basis in x
        c3 = a3
        c2 = a2 - 3a3*x0
        c1 = a1 - 2a2*x0 + 3a3*x0^2
        c0 = a0 - a1*x0 + a2*x0^2 - a3*x0^3

        coeffs[k] = (c0, c1, c2, c3)
    end

    return knots, coeffs
end

# -----------------------------------------------------------------------------
# Exact integral of spline against G_b
# -----------------------------------------------------------------------------

"""
    Gb_times_piecewise_cubic_integral(knots, coeffs, z, H, ψ, κ)

Compute exactly

    ∫_{-H}^0 G_b(z; ξ) B(ξ) dξ

where B(ξ) is piecewise cubic with interval coefficients `coeffs[k]`.
"""
function Gb_times_piecewise_cubic_integral(knots, coeffs, z, H, ψ, κ)
    @assert length(knots) == length(coeffs) + 1
    s = 0.0
    @inbounds for k in 1:length(coeffs)
        a = knots[k]
        b = knots[k+1]
        s += interval_contrib(coeffs[k], a, b, z, H, ψ, κ)
    end
    return s
end

"""
    interval_contrib(c, a, b, z, H, ψ, κ)

Exact contribution of one spline interval [a,b] to

    ∫ G_b(z; ξ) B(ξ) dξ

assuming B(ξ) = c0 + c1 ξ + c2 ξ² + c3 ξ³ on [a,b].
"""
function interval_contrib(c::NTuple{4,<:Real}, a::Real, b::Real,
                          z::Real, H::Real, ψ::Real, κ::Real)
    if b <= z
        A, B = Gb_coeffs(z, H, ψ, κ, :lt)
        return A * polyexp_int(c,  ψ, a, b) +
               B * polyexp_int(c, -ψ, a, b)

    elseif a >= z
        A, B = Gb_coeffs(z, H, ψ, κ, :gt)
        return A * polyexp_int(c,  ψ, a, b) +
               B * polyexp_int(c, -ψ, a, b)

    else
        return interval_contrib(c, a, z, z, H, ψ, κ) +
               interval_contrib(c, z, b, z, H, ψ, κ)
    end
end

# -----------------------------------------------------------------------------
# Final buoyancy profile function
# -----------------------------------------------------------------------------

"""
    buoyancy_profile_from_spline(spl, z, H, ψ, Γ, κ, Ek, b_s)

Compute the buoyancy profile

    b(z) = -(Γ/Ek) * ∫ G_b(z; ξ) B(ξ) dξ + ∂_ξ G_b(z;0) * b_s

where `B(ξ)` is given by BSplineKit spline `spl`.

Returns the scalar buoyancy at depth `z`.
"""
function buoyancy_profile_from_spline(spl, z, H, ψ, Γ, κ, Ek, b_s)
    knots, coeffs = spline_interval_power_coeffs(spl)
    I = Gb_times_piecewise_cubic_integral(knots, coeffs, z, H, ψ, κ)
    surf = dGb_dξ_at_zero(z, H, ψ, κ) * b_s
    return -(Γ / Ek) * I + surf
end