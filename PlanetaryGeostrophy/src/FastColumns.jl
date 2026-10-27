# Fast single-integral column evaluations of the linear PG solution.
#
# The B23 path nests an adaptive z-quadrature around several adaptive
# ξ-quadratures of the large symbolic G_u,b kernels. The functions here use
# the single-integral forms instead: transport forcing (61), bottom-stress
# forcing (58), and horizontal velocity (55). Each needs only the baroclinic
# pressure Π(ξ) = ∫_ξ^0 b dz, which comes from the closed-form kernels
# (A5)--(A6). All vertical integrals use a fixed composite Gauss--Legendre
# rule, so finite differences of Π between neighboring columns are smooth.

using QuadGK: gauss
using Gridap.Arrays: return_cache, evaluate!
using Gridap.CellData: Interpolable, KDTreeSearch

const VFP = VelocityFieldPrototype

"""
    ColumnQuadrature(; points=8, panel_scale=4.0,
                       chebyshev_points_per_scale=8.0,
                       min_chebyshev_points=17, max_chebyshev_points=257)

Fixed composite Gauss--Legendre rule for vertical integrals. A column of depth
`H` is split into `ceil(H / (panel_scale * δ))` equal panels of `points` nodes,
where `δ` is the smaller of the Ekman depth `sqrt(2νEk/|f|)` and the buoyancy
diffusion depth `sqrt(κEk/Γ)`. Integrals with a kernel kink at an interior
level are split there.

Velocity columns interpolate `(∂x + i∂y)Π(ξ)` from Chebyshev--Lobatto nodes.
Their number is `chebyshev_points_per_scale * H / δ + 1`, clamped to the given
limits.
"""
struct ColumnQuadrature
    nodes::Vector{Float64}
    weights::Vector{Float64}
    panel_scale::Float64
    chebyshev_points_per_scale::Float64
    min_chebyshev_points::Int
    max_chebyshev_points::Int
end

function ColumnQuadrature(;
    points=8,
    panel_scale=4.0,
    chebyshev_points_per_scale=8.0,
    min_chebyshev_points=17,
    max_chebyshev_points=257,
)
    points >= 2 || throw(ArgumentError("points must be at least 2"))
    panel_scale > 0 || throw(ArgumentError("panel_scale must be positive"))
    min_chebyshev_points >= 3 ||
        throw(ArgumentError("min_chebyshev_points must be at least 3"))
    max_chebyshev_points >= min_chebyshev_points ||
        throw(ArgumentError("max_chebyshev_points must be ≥ min_chebyshev_points"))
    nodes, weights = gauss(points)
    return ColumnQuadrature(
        nodes, weights, float(panel_scale), float(chebyshev_points_per_scale),
        min_chebyshev_points, max_chebyshev_points,
    )
end

column_scale(p, f) = min(
    sqrt(2 * p.ν * p.Ek / abs(f)),
    sqrt(p.κ * p.Ek / p.Γ),
)

panel_count(q::ColumnQuadrature, p, f, H) =
    max(1, ceil(Int, H / (q.panel_scale * column_scale(p, f))))

"""
    panel_integral(g, a, b, q, panels)

Integrate `g` over `[a, b]` with `panels` equal Gauss--Legendre panels.
"""
function panel_integral(g, a, b, q::ColumnQuadrature, panels::Int)
    nodes, weights = q.nodes, q.weights
    width = (b - a) / panels
    half = width / 2
    total = weights[1] * g(a + half + half * nodes[1])
    @inbounds for k in 1:panels
        middle = a + (k - 0.5) * width
        for i in eachindex(nodes)
            k == 1 && i == 1 && continue
            total += weights[i] * g(middle + half * nodes[i])
        end
    end
    return half * total
end

"""
    split_counts(a, b, c, panels)

Panel counts on `[a, c]` and `[c, b]` with widths no larger than those of an
unsplit `panels`-panel rule. Computing these once per column and reusing them
for neighboring finite-difference columns keeps the rule identical there.
"""
function split_counts(a, b, c, panels)
    a < c < b || return (panels, 0)
    fraction = (c - a) / (b - a)
    return (
        max(1, ceil(Int, panels * fraction)),
        max(1, ceil(Int, panels * (1 - fraction))),
    )
end

function split_panel_integral(g, a, b, c, q::ColumnQuadrature, left, right)
    (a < c < b && right > 0) ||
        return panel_integral(g, a, b, q, left + right)
    return panel_integral(g, a, c, q, left) + panel_integral(g, c, b, q, right)
end

"""
    column_buoyancy_integral(kernels, p, H, B, bs, x, y, ξ, q, left, right)

Evaluate `Π(ξ) = ∫_ξ^0 b dz` at one column from (B7), using the closed-form
`G_b,1` and `G_b,2` kernels (A5)--(A6) and a single ξ' quadrature.
"""
function column_buoyancy_integral(
    kernels, p, Hfun, B, bs, x, y, ξ, q::ColumnQuadrature, left, right,
)
    H = Hfun(x, y)
    ψvalue = VFP.ψ(p)
    interior = split_panel_integral(
        ξp -> kernels.inner_Gb(ξ, ξp, H, ψvalue, p.κ) * B(x, y, ξp),
        -H, 0.0, ξ, q, left, right,
    )
    return -p.Γ / p.Ek * interior +
           kernels.inner_Gb_xi(ξ, H, ψvalue, p.κ) * bs(x, y)
end

"""
    column_buoyancy_integral_gradient(kernels, p, H, B, bs, x, y, ξ, q, panels, step)

Centered-difference `(∂x + i∂y)Π(ξ)` at fixed level `ξ`, the baroclinic
pressure-gradient forcing in (50). The same quadrature rule is used in all four
neighboring columns.
"""
function column_buoyancy_integral_gradient(
    kernels, p, Hfun, B, bs, x, y, ξ, q::ColumnQuadrature, panels, step,
)
    left, right = split_counts(-Hfun(x, y), 0.0, ξ, panels)
    Π(xx, yy) = column_buoyancy_integral(
        kernels, p, Hfun, B, bs, xx, yy, ξ, q, left, right,
    )
    return (
        (Π(x + step, y) - Π(x - step, y)) +
        im * (Π(x, y + step) - Π(x, y - step))
    ) / (2step)
end

"""
    fast_baroclinic_transport(kernels, p, H, f, B, bs, x, y, q; step)

Baroclinic part of the transport forcing `𝔅` in (61), as one vertical
integral of the closed-form kernel times `(∂x + i∂y)Π(ξ)`.
"""
function fast_baroclinic_transport(
    kernels, p, Hfun, ffun, B, bs, x, y, q::ColumnQuadrature; step=1e-5,
)
    H = Hfun(x, y)
    H > 0 || return 0.0 + 0.0im
    f = ffun(x, y)
    qv = sqrt(im * f / (p.ν * p.Ek))
    panels = panel_count(q, p, f, H)
    eH = exp(qv * H)
    integrand(ξ) =
        exp(-qv * ξ) * (exp(qv * ξ) - eH) * (exp(qv * (ξ + H)) - 1) *
        column_buoyancy_integral_gradient(
            kernels, p, Hfun, B, bs, x, y, ξ, q, panels, step,
        )
    baroclinic = panel_integral(integrand, -H, 0.0, q, panels) / p.Ek
    return -baroclinic / (p.ν * qv^2 * (1 + exp(2 * qv * H)))
end

"""
    fast_bottom_stress_forcing(kernels, p, H, f, τ, B, bs, x, y, q; step)

Bottom-stress forcing `T` in (58), with baroclinic and surface-stress parts.
"""
function fast_bottom_stress_forcing(
    kernels, p, Hfun, ffun, τ, B, bs, x, y, q::ColumnQuadrature; step=1e-5,
)
    H = Hfun(x, y)
    H > 0 || return 0.0 + 0.0im
    f = ffun(x, y)
    qv = sqrt(im * f / (p.ν * p.Ek))
    panels = panel_count(q, p, f, H)
    denominator = 1 + exp(2 * qv * H)
    integrand(ξ) =
        exp(qv * (H - ξ)) * (1 + exp(2 * qv * ξ)) / denominator *
        column_buoyancy_integral_gradient(
            kernels, p, Hfun, B, bs, x, y, ξ, q, panels, step,
        )
    baroclinic = panel_integral(integrand, -H, 0.0, q, panels) / p.Ek
    wind = -2 * exp(qv * H) / denominator * τ(x, y)
    return -baroclinic + wind
end

"""
    fast_buoyancy(p, H, B, bs_value, x, y, z, q, panels)

Buoyancy (45) at one level from the closed-form `G_b`, with the ξ quadrature
split at `z`.
"""
function fast_buoyancy(p, H, B, bs_value, x, y, z, q::ColumnQuadrature, panels)
    ψvalue = VFP.ψ(p)
    left, right = split_counts(-H, 0.0, z, panels)
    interior = split_panel_integral(
        ξp -> VFP.Gb(z, ξp, H, p.κ, ψvalue) * B(x, y, ξp),
        -H, 0.0, z, q, left, right,
    )
    return -p.Γ / p.Ek * interior +
           VFP.surface_buoyancy_kernel(z, H, p.κ, ψvalue) * bs_value
end

"""
    fast_column_integral(kernels, p, H, B, bs, x, y, q, panels)

`Π(-H) = ∫_{-H}^0 b dz`, which converts surface to bottom pressure.
"""
fast_column_integral(kernels, p, Hfun, B, bs, x, y, q::ColumnQuadrature, panels) =
    column_buoyancy_integral(
        kernels, p, Hfun, B, bs, x, y, -Hfun(x, y), q, panels, 0,
    )

chebyshev_count(q::ColumnQuadrature, p, f, H) = clamp(
    ceil(Int, q.chebyshev_points_per_scale * H / column_scale(p, f)) + 1,
    q.min_chebyshev_points,
    q.max_chebyshev_points,
)

"""
    fast_velocity_column(kernels, p, H, f, gradps_value, τ_value, B, bs, x, y, q; step)

Return `z -> u + iv` for one water column from (55),

`Ek 𝔲 = G_𝔲,1(z)(∂x+i∂y)p_s - Ek G_𝔲(z;0)τ - ∫ G_𝔲(z;ξ)(∂x+i∂y)Π(ξ) dξ`,

using the closed-form `G_𝔲` (54). `(∂x+i∂y)Π` is tabulated once per column at
Chebyshev--Lobatto levels and evaluated by barycentric interpolation.
"""
function fast_velocity_column(
    kernels, p, Hfun, ffun, gradps_value, τ_value, B, bs, x, y,
    q::ColumnQuadrature; step=1e-5,
)
    H = Hfun(x, y)
    f = ffun(x, y)
    f != 0 || throw(ArgumentError("f(x, y) must be nonzero; got $f"))
    φvalue = VFP.φ_from_f(p, f)
    panels = panel_count(q, p, f, H)
    n = chebyshev_count(q, p, f, H)
    t = [cos(π * j / (n - 1)) for j in 0:n-1]
    levels = @. (t - 1) * H / 2
    gradients = [
        column_buoyancy_integral_gradient(
            kernels, p, Hfun, B, bs, x, y, ξ, q, panels, step,
        ) for ξ in levels
    ]
    barycentric = [(-1.0)^j * (j == 0 || j == n - 1 ? 0.5 : 1.0) for j in 0:n-1]

    function gradient_at(ξ)
        s = 1 + 2ξ / H
        numerator = zero(ComplexF64)
        denominator = 0.0
        @inbounds for j in eachindex(t)
            difference = s - t[j]
            iszero(difference) && return gradients[j]
            weight = barycentric[j] / difference
            numerator += weight * gradients[j]
            denominator += weight
        end
        return numerator / denominator
    end

    function velocity(z)
        VFP.checked_depth(H, z)
        left, right = split_counts(-H, 0.0, z, panels)
        baroclinic = split_panel_integral(
            ξ -> VFP.Gu(z, ξ, H, p.ν, φvalue) * gradient_at(ξ),
            -H, 0.0, z, q, left, right,
        )
        return (
            VFP.Gu_integral(z, H, p.ν, φvalue) * gradps_value -
            p.Ek * VFP.Gu(z, 0.0, H, p.ν, φvalue) * τ_value -
            baroclinic
        ) / p.Ek
    end
    return velocity
end

"""
    PooledPointEvaluator(field, probe)

Thread-safe repeated point evaluation of a Gridap `CellField`. Plain
`evaluate(field, Point(x, y))` rebuilds the cell-search tree on every call
(about 0.5 ms for a 30k-cell mesh). This keeps one search cache per thread in a
pool, which takes a few microseconds per call.
"""
struct PooledPointEvaluator{F,C}
    field::F
    caches::Channel{C}
end

# Gridap locates a point by checking the cells around its nearest mesh
# vertices. With only the nearest vertex (Gridap's default), points can be
# missed where small coastal elements meet larger interior ones; searching
# several vertices avoids that.
const POINT_SEARCH_VERTICES = 8

function PooledPointEvaluator(
    field, probe;
    copies=Threads.nthreads(),
    num_nearest_vertices=POINT_SEARCH_VERTICES,
)
    searchable = Interpolable(field; searchmethod=KDTreeSearch(; num_nearest_vertices))
    first_cache = return_cache(searchable, probe)
    caches = Channel{typeof(first_cache)}(copies)
    put!(caches, first_cache)
    for _ in 2:copies
        put!(caches, return_cache(searchable, probe))
    end
    return PooledPointEvaluator(field, caches)
end

function (evaluator::PooledPointEvaluator)(point)
    cache = take!(evaluator.caches)
    try
        return evaluate!(cache, evaluator.field, point)
    finally
        put!(evaluator.caches, cache)
    end
end

function interior_probe_point(model)
    coordinates = first(get_cell_coordinates(Triangulation(model)))
    return sum(coordinates) / length(coordinates)
end

"""
    nodal_lookup(f, model, order)

Evaluate `f(x)` once at every node of the order-`order` Lagrangian space on
`model`, in parallel, and return a lookup `x -> f(x)`. The lookup reads the
table for those nodes and falls back to `f` at any other point. Interpolating
several finite-element fields of that order from the lookup then costs one
`f` evaluation per node rather than one per field and per node.
"""
function nodal_lookup(f, model, order)
    V = TestFESpace(model, ReferenceFE(lagrangian, Float64, order); conformity=:H1)
    seen = Set{Tuple{Float64,Float64}}()
    points = Tuple{Float64,Float64}[]
    interpolate_everywhere(V) do x
        key = (Float64(x[1]), Float64(x[2]))
        key in seen || (push!(seen, key); push!(points, key))
        0.0
    end
    first_value = f(points[1])
    values = Vector{typeof(first_value)}(undef, length(points))
    values[1] = first_value
    Threads.@threads for i in 2:length(points)
        values[i] = f(points[i])
    end
    table = Dict(zip(points, values))
    lookup(x) = get(() -> f((Float64(x[1]), Float64(x[2]))), table,
                    (Float64(x[1]), Float64(x[2])))
    return lookup, length(points)
end
