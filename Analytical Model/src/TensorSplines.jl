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
# Extract coefficients of a cubic spline in shifted form
# ----------------------------------------------------------------------------

"""
    spline_interval_power_coeffs(spl; domain=nothing)

Return:
- intervals::Vector{Tuple{Float64,Float64}}
- coeffs::Vector{NTuple{4,Float64}}

where on each interval `(a,b)`,

    spl(ξ) = c0 + c1*ξ + c2*ξ^2 + c3*ξ^3.

If `domain=(lo,hi)` is supplied, intervals are clipped to that domain.
"""
function spline_interval_power_coeffs(spl; domain=nothing)
    s = spl.spline  # unwrap SplineInterpolation -> Spline

    rawknots = collect(s.basis.t)

    d1 = BSplineKit.Derivative(1) * s
    d2 = BSplineKit.Derivative(2) * s
    d3 = BSplineKit.Derivative(3) * s

    intervals = Tuple{Float64,Float64}[]
    coeffs = NTuple{4,Float64}[]

    domlo, domhi = domain === nothing ? (-Inf, Inf) : domain

    for k in 1:length(rawknots)-1
        a0 = rawknots[k]
        b0 = rawknots[k+1]

        # Skip repeated knots.
        b0 <= a0 && continue

        # Clip to requested domain.
        a = max(a0, domlo)
        b = min(b0, domhi)

        b <= a && continue

        # midpoint of clipped interval
        x0 = (a + b) / 2

        # shifted Taylor coefficients around x0
        t0 = s(x0)
        t1 = d1(x0)
        t2 = d2(x0) / 2
        t3 = d3(x0) / 6

        # convert shifted cubic to ordinary power basis in ξ
        c3 = t3
        c2 = t2 - 3t3*x0
        c1 = t1 - 2t2*x0 + 3t3*x0^2
        c0 = t0 - t1*x0 + t2*x0^2 - t3*x0^3

        push!(intervals, (a,b))
        push!(coeffs, (c0,c1,c2,c3))
    end

    return intervals, coeffs
end

# -----------------------------------------------------------------------------
# Exact integral of spline against G_b
# -----------------------------------------------------------------------------

"""
    Gb_times_piecewise_cubic_integral(intervals, coeffs, z, H, ψ, κ)

Compute

    ∫_{-H}^0 G_b(z; ξ) B(ξ) dξ

where B is piecewise cubic on `intervals`.

Assumes `z ∈ [-H,0]`.
"""
function Gb_times_piecewise_cubic_integral(intervals, coeffs, z, H, ψ, κ)
    @assert length(intervals) == length(coeffs)
    @assert -H - 10eps() <= z <= 0 + 10eps()

    lo = -H
    hi = 0.0

    s = 0.0

    @inbounds for n in eachindex(coeffs)
        a0, b0 = intervals[n]

        # Clip interval to physical water column.
        a = max(a0, lo)
        b = min(b0, hi)

        b <= a && continue

        s += interval_contrib(coeffs[n], a, b, z, H, ψ, κ)
    end

    return s
end

# -----------------------------------------------------------------------------
# Final buoyancy profile function
# -----------------------------------------------------------------------------

"""
    buoyancy_column_from_spline(spl, zgrid, H, ψ, Γ, κ, Ek, b_s; verbose=false)

Compute b(z) on `zgrid` for one column.

Uses overflow-safe G_b quadrature.
"""
function buoyancy_column_from_spline(spl, zgrid, H, ψ, Γ, κ, Ek, b_s; verbose=false)
    b = fill(NaN, length(zgrid))

    if !isfinite(H)
        verbose && @warn "H is not finite" H
        return b
    end

    if H <= 0
        verbose && @warn "H must be positive ocean depth" H
        return b
    end

    if !isfinite(b_s)
        verbose && @warn "b_s is not finite" b_s
        return b
    end

    valid = findall(k -> (-H <= zgrid[k] <= 0.0), eachindex(zgrid))

    if isempty(valid)
        verbose && @warn "No zgrid points lie inside water column" H bottom=-H zmin=minimum(zgrid) zmax=maximum(zgrid)
        return b
    end

    if verbose
        @info "Computing buoyancy column" H bottom=-H nvalid=length(valid) ψ Γ κ Ek b_s
    end

    @inbounds for k in valid
        z = zgrid[k]

        I = Gb_times_spline_integral_quadrature(spl, z, H, ψ, κ)
        surf = dGb_dξ_at_zero_stable(z, H, ψ, κ) * b_s

        b[k] = -(Γ / Ek) * I + surf
    end

    return b
end

"""
    pressure_integral_from_b_column(zgrid, bcol, H)

Compute

    Π(z) = ∫_z^0 b(z') dz'

on `zgrid`.

Values outside [-H,0] are NaN.
"""
function pressure_integral_from_b_column(zgrid, bcol, H)
    Π = fill(NaN, length(zgrid))

    valid = findall(k -> isfinite(bcol[k]) && (-H <= zgrid[k] <= 0.0), eachindex(zgrid))

    if length(valid) < 4
        return Π
    end

    zvalid = collect(zgrid[valid])
    bvalid = collect(bcol[valid])

    spl = fit_cubic_spline_1d(zvalid, bvalid)
    A = BSplineKit.integral(spl)
    A0 = A(0.0)

    @inbounds for k in valid
        z = zgrid[k]
        Π[k] = A0 - A(z)
    end

    return Π
end