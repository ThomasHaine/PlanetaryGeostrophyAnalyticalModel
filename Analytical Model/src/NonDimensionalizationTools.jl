# ============================================================================
# Nondimensionalization utilities
# ============================================================================

"""
    OceanScales(; L, H0, f0, N0, ν0, κ0, γ, τ0, ρ0=1026.0)

Container for dimensional scales and derived nondimensional numbers.
"""
struct OceanScales
    L::Float64
    H0::Float64
    f0::Float64
    N0::Float64
    ν0::Float64
    κ0::Float64
    γ::Float64
    τ0::Float64
    ρ0::Float64
    U0::Float64
    p0::Float64
    b0::Float64
    Ek::Float64
    Bu::Float64
    Pr::Float64
    Γ::Float64
    δ::Float64
end

function OceanScales(; L, H0, f0, N0, ν0, κ0, γ, τ0, ρ0=1026.0)
    L  = Float64(L)
    H0 = Float64(H0)
    f0 = Float64(f0)
    N0 = Float64(N0)
    ν0 = Float64(ν0)
    κ0 = Float64(κ0)
    γ  = Float64(γ)
    τ0 = Float64(τ0)
    ρ0 = Float64(ρ0)

    U0 = (N0^2 * H0^2) / (f0 * L)
    p0 = ρ0 * f0 * L * U0
    b0 = N0^2 * H0
    Ek = ν0 / (f0 * H0^2)
    Bu = (N0^2 * H0^2) / (f0^2 * L^2)
    Pr = ν0 / κ0
    Γ  = Pr * γ / f0
    δ  = H0 / L

    return OceanScales(L, H0, f0, N0, ν0, κ0, γ, τ0, ρ0, U0, p0, b0, Ek, Bu, Pr, Γ, δ)
end

# ----------------------------------------------------------------------------
# Scalar conversions
# ----------------------------------------------------------------------------

nd_x(x, S::OceanScales) = x ./ S.L
nd_y(y, S::OceanScales) = y ./ S.L
nd_z(z, S::OceanScales) = z ./ S.H0
nd_t(t, S::OceanScales) = t .* S.U0 ./ S.L

rd_x(x̂, S::OceanScales) = x̂ .* S.L
rd_y(ŷ, S::OceanScales) = ŷ .* S.L
rd_z(ẑ, S::OceanScales) = ẑ .* S.H0
rd_t(t̂, S::OceanScales) = t̂ .* S.L ./ S.U0

nd_u(u, S::OceanScales) = u ./ S.U0
nd_v(v, S::OceanScales) = v ./ S.U0
nd_w(w, S::OceanScales) = w ./ ((S.H0 / S.L) * S.U0)

rd_u(û, S::OceanScales) = û .* S.U0
rd_v(v̂, S::OceanScales) = v̂ .* S.U0
rd_w(ŵ, S::OceanScales) = ŵ .* ((S.H0 / S.L) * S.U0)

nd_p(p, S::OceanScales) = p ./ S.p0
nd_b(b, S::OceanScales) = b ./ S.b0

rd_p(p̂, S::OceanScales) = p̂ .* S.p0
rd_b(b̂, S::OceanScales) = b̂ .* S.b0

nd_f(f, S::OceanScales) = f ./ S.f0
nd_ν(ν, S::OceanScales) = ν ./ S.ν0
nd_κ(κ, S::OceanScales) = κ ./ S.κ0

rd_f(f̂, S::OceanScales) = f̂ .* S.f0
rd_ν(ν̂, S::OceanScales) = ν̂ .* S.ν0
rd_κ(κ̂, S::OceanScales) = κ̂ .* S.κ0

nd_τx(τx, S::OceanScales) = τx ./ S.τ0
nd_τy(τy, S::OceanScales) = τy ./ S.τ0

rd_τx(τ̂x, S::OceanScales) = τ̂x .* S.τ0
rd_τy(τ̂y, S::OceanScales) = τ̂y .* S.τ0

# ----------------------------------------------------------------------------
# Array helpers
# ----------------------------------------------------------------------------

"""
    nondimensionalize_field(F, scale)

Return `F ./ scale`.
"""
nondimensionalize_field(F, scale) = F ./ scale

"""
    redimensionalize_field(F̂, scale)

Return `F̂ .* scale`.
"""
redimensionalize_field(F̂, scale) = F̂ .* scale

"""
    nondimensionalize_coords(x, y, z, S)

Return nondimensional coordinates `(x̂, ŷ, ẑ)`.
"""
function nondimensionalize_coords(x, y, z, S::OceanScales)
    return nd_x(x, S), nd_y(y, S), nd_z(z, S)
end

"""
    redimensionalize_coords(x̂, ŷ, ẑ, S)

Return dimensional coordinates `(x, y, z)`.
"""
function redimensionalize_coords(x̂, ŷ, ẑ, S::OceanScales)
    return rd_x(x̂, S), rd_y(ŷ, S), rd_z(ẑ, S)
end

# ----------------------------------------------------------------------------
# Field scaling
# ----------------------------------------------------------------------------

nondimensionalize_buoyancy(b, S::OceanScales) = b ./ S.b0
redimensionalize_buoyancy(b̂, S::OceanScales) = b̂ .* S.b0

nondimensionalize_bathymetry(H, S::OceanScales) = H ./ S.H0
redimensionalize_bathymetry(Ĥ, S::OceanScales) = Ĥ .* S.H0

nondimensionalize_depth(depth2d, S::OceanScales) = depth2d ./ S.H0
redimensionalize_depth(deptĥ, S::OceanScales) = deptĥ .* S.H0

# ----------------------------------------------------------------------------
# Named tuple helpers
# ----------------------------------------------------------------------------

"""
    nondimensionalize_case(fields, S)

Nondimensionalize a case named tuple.
Returns a new named tuple with nondimensional coordinates and fields.
"""
function nondimensionalize_case(fields, S::OceanScales)
    out = Dict{Symbol, Any}()

    # Coordinates
    if haskey(fields, :x)
        out[:x̂] = nondimensionalize_field(fields.x̂, S.L)
    end
    if haskey(fields, :y)
        out[:ŷ] = nondimensionalize_field(fields.ŷ, S.L)
    end
    if haskey(fields, :z)
        out[:ẑ] = nondimensionalize_field(fields.ẑ, S.H0)
    end
    if haskey(fields, :lon)
        out[:lon] = fields.lon
    end
    if haskey(fields, :lat)
        out[:lat] = fields.lat
    end

    # Raw fields
    if haskey(fields, :B)
        out[:B̂] = nondimensionalize_buoyancy(fields.B, S)
    end
    if haskey(fields, :bs)
        out[:b̂s] = nondimensionalize_buoyancy(fields.bs, S)
    end
    if haskey(fields, :H)
        out[:Ĥ] = nondimensionalize_bathymetry(fields.H, S)
    end
    if haskey(fields, :depth2d)
        out[:depth2d̂] = nondimensionalize_depth(fields.depth2d, S)
    end

    # Filled fields
    if haskey(fields, :Bfill)
        out[:B̂fill] = nondimensionalize_buoyancy(fields.Bfill, S)
    end
    if haskey(fields, :bsfill)
        out[:b̂sfill] = nondimensionalize_buoyancy(fields.bsfill, S)
    end

    # WOA thermodynamic fields are left dimensional for now
    if haskey(fields, :T)
        out[:T] = fields.T
    end
    if haskey(fields, :S)
        out[:S] = fields.S
    end
    if haskey(fields, :ρ)
        out[:ρ] = fields.ρ
    end
    if haskey(fields, :Tfill)
        out[:Tfill] = fields.Tfill
    end
    if haskey(fields, :Sfill)
        out[:Sfill] = fields.Sfill
    end
    if haskey(fields, :ρfill)
        out[:ρfill] = fields.ρfill
    end

    # Grids
    if haskey(fields, :xu)
        out[:x̂u] = nondimensionalize_field(fields.xu, S.L)
    end
    if haskey(fields, :yu)
        out[:ŷu] = nondimensionalize_field(fields.yu, S.L)
    end
    if haskey(fields, :zu)
        out[:ẑu] = nondimensionalize_field(fields.zu, S.H0)
    end
    if haskey(fields, :xu2)
        out[:x̂u2] = nondimensionalize_field(fields.xu2, S.L)
    end
    if haskey(fields, :yu2)
        out[:ŷu2] = nondimensionalize_field(fields.yu2, S.L)
    end

    # Resampled buoyancy fields
    if haskey(fields, :Bu)
        out[:B̂u] = nondimensionalize_buoyancy(fields.Bu, S)
    end
    if haskey(fields, :bsu)
        out[:b̂su] = nondimensionalize_buoyancy(fields.bsu, S)
    end
    if haskey(fields, :Tu)
        out[:T̂u] = fields.Tu
    end
    if haskey(fields, :Su)
        out[:Ŝu] = fields.Su
    end
    if haskey(fields, :ρu)
        out[:ρ̂u] = fields.ρu
    end

    return (; out...)
end

"""
    redimensionalize_case(fieldŝ, S)

Redimensionalize a nondimensional case named tuple.
"""
function redimensionalize_case(fieldŝ, S::OceanScales)
    out = Dict{Symbol, Any}()

    if haskey(fieldŝ, :x̂)
        out[:x] = rd_x(fieldŝ.x̂, S)
    end
    if haskey(fieldŝ, :ŷ)
        out[:y] = rd_y(fieldŝ.ŷ, S)
    end
    if haskey(fieldŝ, :ẑ)
        out[:z] = rd_z(fieldŝ.ẑ, S)
    end
    if haskey(fieldŝ, :lon)
        out[:lon] = fieldŝ.lon
    end
    if haskey(fieldŝ, :lat)
        out[:lat] = fieldŝ.lat
    end

    if haskey(fieldŝ, :B̂)
        out[:B] = redimensionalize_buoyancy(fieldŝ.B̂, S)
    end
    if haskey(fieldŝ, :b̂s)
        out[:bs] = redimensionalize_buoyancy(fieldŝ.b̂s, S)
    end
    if haskey(fieldŝ, :Ĥ)
        out[:H] = redimensionalize_bathymetry(fieldŝ.Ĥ, S)
    end
    if haskey(fieldŝ, :depth2d̂)
        out[:depth2d] = redimensionalize_depth(fieldŝ.depth2d̂, S)
    end
    if haskey(fieldŝ, :B̂fill)
        out[:Bfill] = redimensionalize_buoyancy(fieldŝ.B̂fill, S)
    end
    if haskey(fieldŝ, :b̂sfill)
        out[:bsfill] = redimensionalize_buoyancy(fieldŝ.b̂sfill, S)
    end

    if haskey(fieldŝ, :T)
        out[:T] = fieldŝ.T
    end
    if haskey(fieldŝ, :S)
        out[:S] = fieldŝ.S
    end
    if haskey(fieldŝ, :ρ)
        out[:ρ] = fieldŝ.ρ
    end
    if haskey(fieldŝ, :Tfill)
        out[:Tfill] = fieldŝ.Tfill
    end
    if haskey(fieldŝ, :Sfill)
        out[:Sfill] = fieldŝ.Sfill
    end
    if haskey(fieldŝ, :ρfill)
        out[:ρfill] = fieldŝ.ρfill
    end

    if haskey(fieldŝ, :x̂u)
        out[:xu] = rd_x(fieldŝ.x̂u, S)
    end
    if haskey(fieldŝ, :ŷu)
        out[:yu] = rd_y(fieldŝ.ŷu, S)
    end
    if haskey(fieldŝ, :ẑu)
        out[:zu] = rd_z(fieldŝ.ẑu, S)
    end
    if haskey(fieldŝ, :x̂u2)
        out[:xu2] = rd_x(fieldŝ.x̂u2, S)
    end
    if haskey(fieldŝ, :ŷu2)
        out[:yu2] = rd_y(fieldŝ.ŷu2, S)
    end

    if haskey(fieldŝ, :B̂u)
        out[:Bu] = redimensionalize_buoyancy(fieldŝ.B̂u, S)
    end
    if haskey(fieldŝ, :b̂su)
        out[:bsu] = redimensionalize_buoyancy(fieldŝ.b̂su, S)
    end
    if haskey(fieldŝ, :T̂u)
        out[:Tu] = fieldŝ.T̂u
    end
    if haskey(fieldŝ, :Ŝu)
        out[:Su] = fieldŝ.Ŝu
    end
    if haskey(fieldŝ, :ρ̂u)
        out[:ρu] = fieldŝ.ρ̂u
    end

    return (; out...)
end

# ----------------------------------------------------------------------------
# Rebuild nondimensional interpolants
# ----------------------------------------------------------------------------

"""
    rebuild_nondimensional_case(fields, S)

Nondimensionalize a case and then rebuild spline objects in nondimensional space.
"""
function rebuild_nondimensional_case(fields, S::OceanScales)
    out = Dict{Symbol, Any}()

    # ---------------------------------------------------------------------
    # Coordinates
    # ---------------------------------------------------------------------
    x̂ = haskey(fields, :x) ? nondimensionalize_field(fields.x, S.L) : nothing
    ŷ = haskey(fields, :y) ? nondimensionalize_field(fields.y, S.L) : nothing
    ẑ = haskey(fields, :z) ? nondimensionalize_field(fields.z, S.H0) : nothing

    if x̂ !== nothing
        out[:x̂] = x̂
    end
    if ŷ !== nothing
        out[:ŷ] = ŷ
    end
    if ẑ !== nothing
        out[:ẑ] = ẑ
    end

    # ---------------------------------------------------------------------
    # Nondimensionalize primary fields
    # ---------------------------------------------------------------------
    if haskey(fields, :Bfill)
        out[:B̂fill] = nondimensionalize_buoyancy(fields.Bfill, S)
    elseif haskey(fields, :B)
        out[:B̂fill] = nondimensionalize_buoyancy(fields.B, S)
    end

    if haskey(fields, :bsfill)
        out[:b̂sfill] = nondimensionalize_buoyancy(fields.bsfill, S)
    elseif haskey(fields, :bs)
        out[:b̂sfill] = nondimensionalize_buoyancy(fields.bs, S)
    end

    if haskey(fields, :H)
        out[:Ĥ] = nondimensionalize_bathymetry(fields.H, S)
    end

    # ---------------------------------------------------------------------
    # Rebuild 3D buoyancy spline columns in nondimensional space
    # ---------------------------------------------------------------------
    if haskey(out, :x̂) && haskey(out, :ŷ) && haskey(out, :ẑ) && haskey(out, :B̂fill)
        x̂u, ŷu, ẑu, B̂u, B̂zsplines = build_uniform_bspline_3d_bsplinekit(
            out[:x̂], out[:ŷ], out[:ẑ], out[:B̂fill];
            nx = length(out[:x̂]),
            ny = length(out[:ŷ]),
            nz = length(out[:ẑ])
        )

        out[:x̂u] = x̂u
        out[:ŷu] = ŷu
        out[:ẑu] = ẑu
        out[:B̂u] = B̂u
        out[:B̂zsplines] = B̂zsplines
    end

    # ---------------------------------------------------------------------
    # Rebuild 2D surface buoyancy spline rows in nondimensional space
    # ---------------------------------------------------------------------
    if haskey(out, :x̂) && haskey(out, :ŷ) && haskey(out, :b̂sfill)
        bs_work = fill_nan_neighbors_2d!(copy(out[:b̂sfill]))

        x̂u2, ŷu2, b̂su, b̂ssplines = build_uniform_bspline_2d_bsplinekit(
            out[:x̂], out[:ŷ], bs_work;
            nx = length(out[:x̂]),
            ny = length(out[:ŷ])
        )

        out[:x̂u2] = x̂u2
        out[:ŷu2] = ŷu2
        out[:b̂su] = b̂su
        out[:b̂ssplines] = b̂ssplines
    end

    # ---------------------------------------------------------------------
    # Rebuild 2D bathymetry spline if needed
    # ---------------------------------------------------------------------
    if haskey(out, :x̂) && haskey(out, :ŷ) && haskey(out, :Ĥ)
        H_work = fill_nan_neighbors_2d!(copy(out[:Ĥ]))

        x̂h, ŷh, Ĥf, Ĥsplines = build_uniform_bspline_2d_bsplinekit(
            out[:x̂], out[:ŷ], H_work;
            nx = length(out[:x̂]),
            ny = length(out[:ŷ])
        )

        out[:x̂h] = x̂h
        out[:ŷh] = ŷh
        out[:Ĥf] = Ĥf
        out[:Ĥsplines] = Ĥsplines
    end

    return (; out...)
end

# ----------------------------------------------------------------------------
# Nondimensional plotting helpers
# ----------------------------------------------------------------------------

"""
    plot_ocean_depth_nd(deptĥ, x̂, ŷ; xlim=[-1, 1], ylim=[-1, 1])

Plot nondimensional ocean depth.
"""
function plot_ocean_depth_nd(deptĥ, x̂, ŷ; xlim=[-1, 1], ylim=[-1, 1])
    dep = copy(deptĥ)
    dep[.!isfinite.(dep) .| (dep .<= 0)] .= NaN
    vals = dep[isfinite.(dep)]
    @assert !isempty(vals) "No finite nondimensional depth values to plot."

    p = heatmap(
        x̂, ŷ, dep',
        xlim = xlim, ylim = ylim,
        xlabel = "x̂",
        ylabel = "ŷ",
        title = "Nondimensional ocean depth",
        color = :viridis,
        colorbar_title = "Ĥ",
        clims = (minimum(vals), maximum(vals))
    )

    return p
end

"""
    plot_surface_buoyancy_nd(b̂s, x̂u, ŷu, deptĥ; xlim=[-1, 1], ylim=[-1, 1])

Plot nondimensional surface buoyancy, masking land.
"""
function plot_surface_buoyancy_nd(b̂s, x̂u, ŷu, deptĥ;
                                  xlim=[-1, 1], ylim=[-1, 1])
    A = mask_land(b̂s, deptĥ)

    vals = A[isfinite.(A)]
    @assert !isempty(vals) "No finite nondimensional surface buoyancy values."

    p = heatmap(
        x̂u, ŷu, A',
        xlim = xlim, ylim = ylim,
        xlabel = "x̂",
        ylabel = "ŷ",
        title = "Nondimensional surface buoyancy",
        color = :balance,
        aspect_ratio = :equal,
        colorbar_title = "b̂s",
        clims = (minimum(vals), maximum(vals))
    )
    return p
end

"""
    plot_vertical_profile_comparison_nd(x̂0, ŷ0, ẑ, B̂, B̂zsplines, x̂grid, ŷgrid)

Plot a nondimensional buoyancy profile comparison.
"""
function plot_vertical_profile_comparison_nd(x̂0, ŷ0, ẑ, B̂, B̂zsplines, x̂grid, ŷgrid)
    ix = argmin(abs.(x̂grid .- x̂0))
    iy = argmin(abs.(ŷgrid .- ŷ0))

    trueB   = [B̂[ix, iy, k] for k in 1:length(ẑ)]
    approxB = [B̂zsplines[ix, iy](zk) for zk in ẑ]

    ok = isfinite.(trueB) .& isfinite.(approxB)

    p = plot(
        trueB[ok], ẑ[ok],
        label = "Original",
        linewidth = 2,
        xlabel = "b̂",
        ylabel = "ẑ",
        title = "Nondimensional buoyancy profile",
        legend = :bottomright
    )
    plot!(p, approxB[ok], ẑ[ok], label = "Spline", linewidth = 2, linestyle = :dash)
    return p
end