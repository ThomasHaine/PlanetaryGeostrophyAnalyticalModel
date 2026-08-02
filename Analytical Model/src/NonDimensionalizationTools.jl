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

Nondimensionalize a `fields` named tuple from either a WOA or ideal case.

Returns a new named tuple with nondimensional coordinates and fields.
"""
function nondimensionalize_case(fields, S::OceanScales)
    out = Dict{Symbol, Any}()

    # Coordinates
    if haskey(fields, :x)
        out[:x] = nondimensionalize_field(fields.x, S.L)
    end
    if haskey(fields, :y)
        out[:y] = nondimensionalize_field(fields.y, S.L)
    end
    if haskey(fields, :z)
        out[:z] = nondimensionalize_field(fields.z, S.H0)
    end
    if haskey(fields, :lon)
        out[:lon] = fields.lon
    end
    if haskey(fields, :lat)
        out[:lat] = fields.lat
    end

    # Raw fields
    if haskey(fields, :B)
        out[:B] = nondimensionalize_buoyancy(fields.B, S)
    end
    if haskey(fields, :bs)
        out[:bs] = nondimensionalize_buoyancy(fields.bs, S)
    end
    if haskey(fields, :H)
        out[:H] = nondimensionalize_bathymetry(fields.H, S)
    end
    if haskey(fields, :depth2d)
        out[:depth2d] = nondimensionalize_depth(fields.depth2d, S)
    end

    # Filled fields
    if haskey(fields, :Bfill)
        out[:Bfill] = nondimensionalize_buoyancy(fields.Bfill, S)
    end
    if haskey(fields, :bsfill)
        out[:bsfill] = nondimensionalize_buoyancy(fields.bsfill, S)
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
        out[:xu] = nondimensionalize_field(fields.xu, S.L)
    end
    if haskey(fields, :yu)
        out[:yu] = nondimensionalize_field(fields.yu, S.L)
    end
    if haskey(fields, :zu)
        out[:zu] = nondimensionalize_field(fields.zu, S.H0)
    end
    if haskey(fields, :xu2)
        out[:xu2] = nondimensionalize_field(fields.xu2, S.L)
    end
    if haskey(fields, :yu2)
        out[:yu2] = nondimensionalize_field(fields.yu2, S.L)
    end

    # Resampled buoyancy fields
    if haskey(fields, :Bu)
        out[:Bu] = nondimensionalize_buoyancy(fields.Bu, S)
    end
    if haskey(fields, :bsu)
        out[:bsu] = nondimensionalize_buoyancy(fields.bsu, S)
    end

    # Interpolants are preserved; they are still dimensional unless rebuilt
    if haskey(fields, :Bsp)
        out[:Bsp] = fields.Bsp
    end
    if haskey(fields, :bsp)
        out[:bsp] = fields.bsp
    end
    if haskey(fields, :Tsp)
        out[:Tsp] = fields.Tsp
    end
    if haskey(fields, :Ssp)
        out[:Ssp] = fields.Ssp
    end
    if haskey(fields, :ρsp)
        out[:ρsp] = fields.ρsp
    end

    return (; out...)
end

"""
    redimensionalize_case(fieldŝ, S)

Redimensionalize a nondimensional case named tuple.
"""
function redimensionalize_case(fieldŝ, S::OceanScales)
    out = Dict{Symbol, Any}()

    if haskey(fieldŝ, :x)
        out[:x] = rd_x(fieldŝ.x, S)
    end
    if haskey(fieldŝ, :y)
        out[:y] = rd_y(fieldŝ.y, S)
    end
    if haskey(fieldŝ, :z)
        out[:z] = rd_z(fieldŝ.z, S)
    end
    if haskey(fieldŝ, :lon)
        out[:lon] = fieldŝ.lon
    end
    if haskey(fieldŝ, :lat)
        out[:lat] = fieldŝ.lat
    end

    if haskey(fieldŝ, :B)
        out[:B] = redimensionalize_buoyancy(fieldŝ.B, S)
    end
    if haskey(fieldŝ, :bs)
        out[:bs] = redimensionalize_buoyancy(fieldŝ.bs, S)
    end
    if haskey(fieldŝ, :H)
        out[:H] = redimensionalize_bathymetry(fieldŝ.H, S)
    end
    if haskey(fieldŝ, :depth2d)
        out[:depth2d] = redimensionalize_depth(fieldŝ.depth2d, S)
    end
    if haskey(fieldŝ, :Bfill)
        out[:Bfill] = redimensionalize_buoyancy(fieldŝ.Bfill, S)
    end
    if haskey(fieldŝ, :bsfill)
        out[:bsfill] = redimensionalize_buoyancy(fieldŝ.bsfill, S)
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

    if haskey(fieldŝ, :xu)
        out[:xu] = rd_x(fieldŝ.xu, S)
    end
    if haskey(fieldŝ, :yu)
        out[:yu] = rd_y(fieldŝ.yu, S)
    end
    if haskey(fieldŝ, :zu)
        out[:zu] = rd_z(fieldŝ.zu, S)
    end
    if haskey(fieldŝ, :xu2)
        out[:xu2] = rd_x(fieldŝ.xu2, S)
    end
    if haskey(fieldŝ, :yu2)
        out[:yu2] = rd_y(fieldŝ.yu2, S)
    end

    if haskey(fieldŝ, :Bu)
        out[:Bu] = redimensionalize_buoyancy(fieldŝ.Bu, S)
    end
    if haskey(fieldŝ, :bsu)
        out[:bsu] = redimensionalize_buoyancy(fieldŝ.bsu, S)
    end

    if haskey(fieldŝ, :Bsp)
        out[:Bsp] = fieldŝ.Bsp
    end
    if haskey(fieldŝ, :bsp)
        out[:bsp] = fieldŝ.bsp
    end
    if haskey(fieldŝ, :Tsp)
        out[:Tsp] = fieldŝ.Tsp
    end
    if haskey(fieldŝ, :Ssp)
        out[:Ssp] = fieldŝ.Ssp
    end
    if haskey(fieldŝ, :ρsp)
        out[:ρsp] = fieldŝ.ρsp
    end

    return (; out...)
end

# ----------------------------------------------------------------------------
# Rebuild nondimensional interpolants
# ----------------------------------------------------------------------------

function rebuild_nondimensional_case(fields, S::OceanScales)
    out = Dict{Symbol, Any}()

    # ---------------------------------------------------------------------
    # Coordinates
    # ---------------------------------------------------------------------
    x̂ = haskey(fields, :x) ? nondimensionalize_field(fields.x, S.L) : nothing
    ŷ = haskey(fields, :y) ? nondimensionalize_field(fields.y, S.L) : nothing
    ẑ = haskey(fields, :z) ? nondimensionalize_field(fields.z, S.H0) : nothing

    if x̂ !== nothing
        out[:x] = x̂
    end
    if ŷ !== nothing
        out[:y] = ŷ
    end
    if ẑ !== nothing
        out[:z] = ẑ
    end

    # ---------------------------------------------------------------------
    # Nondimensionalize primary fields
    # ---------------------------------------------------------------------
    if haskey(fields, :Bfill)
        out[:Bfill] = nondimensionalize_buoyancy(fields.Bfill, S)
    elseif haskey(fields, :B)
        out[:Bfill] = nondimensionalize_buoyancy(fields.B, S)
    end

    if haskey(fields, :bsfill)
        out[:bsfill] = nondimensionalize_buoyancy(fields.bsfill, S)
    elseif haskey(fields, :bs)
        out[:bsfill] = nondimensionalize_buoyancy(fields.bs, S)
    end

    if haskey(fields, :H)
        out[:H] = nondimensionalize_bathymetry(fields.H, S)
    end

    # ---------------------------------------------------------------------
    # Rebuild 3D buoyancy spline: Bsp
    # ---------------------------------------------------------------------
    if haskey(out, :x) && haskey(out, :y) && haskey(out, :z) && haskey(out, :Bfill)
        xu, yu, zu, Bu, Bsp = build_uniform_bspline_3d(
            out[:x], out[:y], out[:z], out[:Bfill];
            nx = length(out[:x]),
            ny = length(out[:y]),
            nz = length(out[:z])
        )

        out[:xu] = xu
        out[:yu] = yu
        out[:zu] = zu
        out[:Bu] = Bu
        out[:Bsp] = Bsp
    end

    # ---------------------------------------------------------------------
    # Rebuild 2D surface buoyancy spline: bsp
    # ---------------------------------------------------------------------
    if haskey(out, :x) && haskey(out, :y) && haskey(out, :bsfill)
        bs_work = copy(out[:bsfill])
        bs_work = fill_nan_neighbors_2d!(bs_work)

        xus, yus, Bsu, bsp = build_uniform_bspline_2d(
            out[:x], out[:y], bs_work;
            nx = length(out[:x]),
            ny = length(out[:y])
        )

        out[:xus] = xus
        out[:yus] = yus
        out[:bsu] = Bsu
        out[:bsp] = bsp
    end

    # ---------------------------------------------------------------------
    # Rebuild 2D bathymetry/depth spline: Hsp
    # ---------------------------------------------------------------------
    if haskey(out, :x) && haskey(out, :y) && haskey(out, :H)
        H_work = copy(out[:H])
        H_work = fill_nan_neighbors_2d!(H_work)

        xh, yh, Hf, Hsp = build_uniform_bspline_2d(
            out[:x], out[:y], H_work;
            nx = length(out[:x]),
            ny = length(out[:y])
        )

        out[:xh] = xh
        out[:yh] = yh
        out[:Hf] = Hf
        out[:Hsp] = Hsp
    end

    return (; out...)
end

# ----------------------------------------------------------------------------
# Nondimensional plotting helpers
# ----------------------------------------------------------------------------

"""
    plot_ocean_depth_nd(deptĥ, x̂, ŷ; xlim=[-1, 1], ylim=[-1, 1])

Plot nondimensional ocean depth.
"""
function plot_ocean_depth_nd(deptĥ, x̂, ŷ; xlim=[-1, 1], ylim=[-1, 1])
    dep = copy(deptĥ)
    dep[.!isfinite.(dep) .| (dep .<= 0)] .= NaN
    vals = dep[isfinite.(dep)]
    @assert !isempty(vals) "No finite nondimensional depth values to plot."

    p = heatmap(
        x̂, ŷ, dep',
        xlim = xlim, ylim = ylim,
        xlabel = "x̂",
        ylabel = "ŷ",
        title = "Nondimensional ocean depth",
        color = :viridis,
        colorbar_title = "Ĥ",
        clims = (minimum(vals), maximum(vals))
    )

    return p
end

"""
    plot_surface_buoyancy_nd(B̂sp, x̂u, ŷu, deptĥ; xlim=[-1, 1], ylim=[-1, 1])

Plot nondimensional surface buoyancy, masking land.
"""
function plot_surface_buoyancy_nd(bs, x̂u, ŷu, deptĥ;
                                  xlim=[-1, 1], ylim=[-1, 1])
    A = mask_land(bs, deptĥ)

    vals = A[isfinite.(A)]
    @assert !isempty(vals) "No finite nondimensional surface buoyancy values."

    p = heatmap(
        x̂u, ŷu, A',
        xlim = xlim, ylim = ylim,
        xlabel = "x̂",
        ylabel = "ŷ",
        title = "Nondimensional surface buoyancy",
        color = :balance,
        aspect_ratio = :equal,
        colorbar_title = "b̂",
        clims = (minimum(vals), maximum(vals))
    )
    return p
end

"""
    plot_vertical_profile_comparison_nd(x̂0, ŷ0, ẑ, B̂, B̂sp, x̂grid, ŷgrid)

Plot a nondimensional buoyancy profile comparison.
"""
function plot_vertical_profile_comparison_nd(x̂0, ŷ0, ẑ, B̂, B̂sp, x̂grid, ŷgrid)
    ix = argmin(abs.(x̂grid .- x̂0))
    iy = argmin(abs.(ŷgrid .- ŷ0))

    trueB   = [B̂[ix, iy, k] for k in 1:length(ẑ)]
    approxB = [B̂sp(x̂0, ŷ0, zk) for zk in ẑ]

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