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
# Interpolation builders
# ============================================================================

function build_uniform_bspline_3d(x, y, z, F; nx=length(x), ny=length(y), nz=length(z),
                                  xuniform=nothing, yuniform=nothing, zuniform=nothing)
    @assert size(F) == (length(x), length(y), length(z))
    @assert all(isfinite, F)
    @assert issorted(x) && issorted(y) && issorted(z)

    tmp_itp = interpolate((x, y, z), F, Gridded(Linear()))

    xu = xuniform === nothing ? range(first(x), last(x), length=nx) : xuniform
    yu = yuniform === nothing ? range(first(y), last(y), length=ny) : yuniform
    zu = zuniform === nothing ? range(first(z), last(z), length=nz) : zuniform

    Fu = Array{Float64}(undef, length(xu), length(yu), length(zu))
    @inbounds for i in eachindex(xu), j in eachindex(yu), k in eachindex(zu)
        Fu[i, j, k] = tmp_itp(xu[i], yu[j], zu[k])
    end

    itp = interpolate(Fu, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, xu, yu, zu)
    return xu, yu, zu, Fu, sitp
end

function build_uniform_bspline_2d(x, y, F; nx=length(x), ny=length(y),
                                  xuniform=nothing, yuniform=nothing)
    @assert size(F) == (length(x), length(y))
    @assert all(isfinite, F)
    @assert issorted(x) && issorted(y)

    tmp_itp = interpolate((x, y), F, Gridded(Linear()))

    xu = xuniform === nothing ? range(first(x), last(x), length=nx) : xuniform
    yu = yuniform === nothing ? range(first(y), last(y), length=ny) : yuniform

    Fu = Array{Float64}(undef, length(xu), length(yu))
    @inbounds for i in eachindex(xu), j in eachindex(yu)
        Fu[i, j] = tmp_itp(xu[i], yu[j])
    end

    itp = interpolate(Fu, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, xu, yu)
    return xu, yu, Fu, sitp
end

function interp2d_on_grid(F, x, y, xnew, ynew)
    itp = interpolate((x, y), F, Gridded(Linear()))
    G = Array{Float64}(undef, length(xnew), length(ynew))
    for i in eachindex(xnew), j in eachindex(ynew)
        G[i,j] = itp(xnew[i], ynew[j])
    end
    return G
end