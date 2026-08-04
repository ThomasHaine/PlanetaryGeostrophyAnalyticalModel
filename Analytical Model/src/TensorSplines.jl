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