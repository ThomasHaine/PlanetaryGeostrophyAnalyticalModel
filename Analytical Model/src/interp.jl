# ============================================================================
# NaN filling
# ============================================================================

function fill_nan_neighbors!(A; maxiter=100, radius=2)
    A = copy(A)
    nx, ny, nz = size(A)

    for _ in 1:maxiter
        changed = false
        B = copy(A)
        @inbounds for i in 1:nx, j in 1:ny, k in 1:nz
            if !isfinite(A[i,j,k])
                vals = Float64[]
                for di in -radius:radius, dj in -radius:radius, dk in -radius:radius
                    ii, jj, kk = i+di, j+dj, k+dk
                    if 1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz && isfinite(A[ii,jj,kk])
                        push!(vals, A[ii,jj,kk])
                    end
                end
                if !isempty(vals)
                    B[i,j,k] = mean(vals)
                    changed = true
                end
            end
        end
        A = B
        !changed && break
    end
    return A
end

function fill_nan_neighbors_2d!(A; maxiter=100, radius=2)
    A = copy(A)
    nx, ny = size(A)

    for _ in 1:maxiter
        changed = false
        B = copy(A)
        @inbounds for i in 1:nx, j in 1:ny
            if !isfinite(A[i,j])
                vals = Float64[]
                for di in -radius:radius, dj in -radius:radius
                    ii, jj = i+di, j+dj
                    if 1 <= ii <= nx && 1 <= jj <= ny && isfinite(A[ii,jj])
                        push!(vals, A[ii,jj])
                    end
                end
                if !isempty(vals)
                    B[i,j] = mean(vals)
                    changed = true
                end
            end
        end
        A = B
        !changed && break
    end
    return A
end

# ============================================================================
# BSplineKit utilities
# ============================================================================

using BSplineKit
using Statistics

"""
    fit_cubic_spline_1d(x, f)

Fit a cubic spline to samples `f` on grid `x`.

Returns the spline object.
"""
function fit_cubic_spline_1d(x, f)
    x = collect(x)
    f = collect(f)

    @assert length(x) == length(f)
    @assert issorted(x)
    @assert all(isfinite, f)

    spline = interpolate(x, f, BSplineOrder(4))
    return spline
end

"""
    spline_definite_integral(spline, a, b)

Compute ∫ₐᵇ spline(x) dx analytically.
"""
function spline_definite_integral(spline, a, b)
    F = BSplineKit.Splines.integral(spline)
    return F(b) - F(a)
end

# ============================================================================
# Manual interpolation helpers (no Interpolations.jl)
# ============================================================================

function lininterp1(x::AbstractVector, f::AbstractVector, xq::Real)
    @assert issorted(x)
    n = length(x)

    if xq <= x[1]
        return f[1]
    elseif xq >= x[end]
        return f[end]
    end

    i = searchsortedlast(x, xq)
    i = clamp(i, 1, n-1)

    x1, x2 = x[i], x[i+1]
    f1, f2 = f[i], f[i+1]
    t = (xq - x1) / (x2 - x1)
    return (1 - t)*f1 + t*f2
end

function bilinterp2(x::AbstractVector, y::AbstractVector, F::AbstractMatrix,
                    xq::Real, yq::Real)
    @assert size(F) == (length(x), length(y))
    @assert issorted(x) && issorted(y)

    if xq <= x[1]
        i = 1
    elseif xq >= x[end]
        i = length(x)-1
    else
        i = clamp(searchsortedlast(x, xq), 1, length(x)-1)
    end

    if yq <= y[1]
        j = 1
    elseif yq >= y[end]
        j = length(y)-1
    else
        j = clamp(searchsortedlast(y, yq), 1, length(y)-1)
    end

    x1, x2 = x[i], x[i+1]
    y1, y2 = y[j], y[j+1]

    tx = (xq - x1) / (x2 - x1)
    ty = (yq - y1) / (y2 - y1)

    f11 = F[i,   j]
    f21 = F[i+1, j]
    f12 = F[i,   j+1]
    f22 = F[i+1, j+1]

    return (1-tx)*(1-ty)*f11 + tx*(1-ty)*f21 + (1-tx)*ty*f12 + tx*ty*f22
end

function trilinterp3(x::AbstractVector, y::AbstractVector, z::AbstractVector,
                     F::Array, xq::Real, yq::Real, zq::Real)
    @assert size(F) == (length(x), length(y), length(z))
    @assert issorted(x) && issorted(y) && issorted(z)

    ix = xq <= x[1] ? 1 : (xq >= x[end] ? length(x)-1 : clamp(searchsortedlast(x, xq), 1, length(x)-1))
    iy = yq <= y[1] ? 1 : (yq >= y[end] ? length(y)-1 : clamp(searchsortedlast(y, yq), 1, length(y)-1))
    iz = zq <= z[1] ? 1 : (zq >= z[end] ? length(z)-1 : clamp(searchsortedlast(z, zq), 1, length(z)-1))

    x1, x2 = x[ix], x[ix+1]
    y1, y2 = y[iy], y[iy+1]
    z1, z2 = z[iz], z[iz+1]

    tx = (xq - x1) / (x2 - x1)
    ty = (yq - y1) / (y2 - y1)
    tz = (zq - z1) / (z2 - z1)

    c000 = F[ix,   iy,   iz]
    c100 = F[ix+1, iy,   iz]
    c010 = F[ix,   iy+1, iz]
    c110 = F[ix+1, iy+1, iz]
    c001 = F[ix,   iy,   iz+1]
    c101 = F[ix+1, iy,   iz+1]
    c011 = F[ix,   iy+1, iz+1]
    c111 = F[ix+1, iy+1, iz+1]

    c00 = (1-tx)*c000 + tx*c100
    c10 = (1-tx)*c010 + tx*c110
    c01 = (1-tx)*c001 + tx*c101
    c11 = (1-tx)*c011 + tx*c111

    c0 = (1-ty)*c00 + ty*c10
    c1 = (1-ty)*c01 + ty*c11

    return (1-tz)*c0 + tz*c1
end

function resample_to_uniform_2d(x, y, F; nx=length(x), ny=length(y),
                                xuniform=nothing, yuniform=nothing)
    xu = xuniform === nothing ? range(first(x), last(x), length=nx) : xuniform
    yu = yuniform === nothing ? range(first(y), last(y), length=ny) : yuniform

    Fu = Array{Float64}(undef, length(xu), length(yu))
    @inbounds for i in eachindex(xu), j in eachindex(yu)
        Fu[i, j] = bilinterp2(x, y, F, xu[i], yu[j])
    end

    return xu, yu, Fu
end

function resample_to_uniform_3d(x, y, z, F; nx=length(x), ny=length(y), nz=length(z),
                                xuniform=nothing, yuniform=nothing, zuniform=nothing)
    xu = xuniform === nothing ? range(first(x), last(x), length=nx) : xuniform
    yu = yuniform === nothing ? range(first(y), last(y), length=ny) : yuniform
    zu = zuniform === nothing ? range(first(z), last(z), length=nz) : zuniform

    Fu = Array{Float64}(undef, length(xu), length(yu), length(zu))
    @inbounds for i in eachindex(xu), j in eachindex(yu), k in eachindex(zu)
        Fu[i, j, k] = trilinterp3(x, y, z, F, xu[i], yu[j], zu[k])
    end

    return xu, yu, zu, Fu
end

# ============================================================================
# Spline builders
# ============================================================================

"""
    build_vertical_splines(x, y, z, F)

Build a cubic spline in z for each (x,y) column of 3D field `F`.

Returns:
- `spline_cols[i,j]` : spline object for F[i,j,:]
"""
function build_vertical_splines(x, y, z, F)
    @assert size(F) == (length(x), length(y), length(z))
    @assert all(isfinite, F)
    @assert issorted(z)

    spline_cols = Matrix{Any}(undef, length(x), length(y))
    @inbounds for i in eachindex(x), j in eachindex(y)
        spline_cols[i, j] = fit_cubic_spline_1d(z, vec(F[i, j, :]))
    end

    return spline_cols
end

"""
    build_surface_splines(x, y, bs)

Build a cubic spline in y for each x-row of `bs`.

Returns:
- `spline_rows[i]` : spline object for bs[i,:]
"""
function build_surface_splines(x, y, bs)
    @assert size(bs) == (length(x), length(y))
    @assert all(isfinite, bs)

    spline_rows = Vector{Any}(undef, length(x))
    @inbounds for i in eachindex(x)
        spline_rows[i] = fit_cubic_spline_1d(y, vec(bs[i, :]))
    end

    return spline_rows
end

# ============================================================================
# High-level builders
# ============================================================================

function build_uniform_gridded_field_3D(x, y, z, F;
                                             nx=length(x), ny=length(y), nz=length(z),
                                             xuniform=nothing, yuniform=nothing, zuniform=nothing)
    @assert size(F) == (length(x), length(y), length(z))
    @assert all(isfinite, F)
    @assert issorted(x) && issorted(y) && issorted(z)

    xu, yu, zu, Fu = resample_to_uniform_3d(x, y, z, F;
                                            nx=nx, ny=ny, nz=nz,
                                            xuniform=xuniform, yuniform=yuniform, zuniform=zuniform)

    return xu, yu, zu, Fu
end

function build_uniform_gridded_field_2D(x, y, F;
                                             nx=length(x), ny=length(y),
                                             xuniform=nothing, yuniform=nothing)
    @assert size(F) == (length(x), length(y))
    @assert all(isfinite, F)
    @assert issorted(x) && issorted(y)

    xu, yu, Fu = resample_to_uniform_2d(x, y, F;
                                        nx=nx, ny=ny,
                                        xuniform=xuniform, yuniform=yuniform)

    return xu, yu, Fu
end

# ============================================================================
# Direct evaluation helpers on grids
# ============================================================================

function interp2d_on_grid(F, x, y, xnew, ynew)
    G = Array{Float64}(undef, length(xnew), length(ynew))
    for i in eachindex(xnew), j in eachindex(ynew)
        G[i,j] = bilinterp2(x, y, F, xnew[i], ynew[j])
    end
    return G
end