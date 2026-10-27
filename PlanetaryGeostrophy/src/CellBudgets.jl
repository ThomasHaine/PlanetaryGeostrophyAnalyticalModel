# Cell-averaged (control-volume) vorticity budgets.
#
# The curls and Jacobians in the BV (27), DIV (32), GCVB (34), and TDVB (37)
# balances act on fields that contain the finite-element surface-pressure
# gradient. That gradient is discontinuous across element edges, so pointwise
# or grid finite differences of the bottom stress or transport are not
# convergent. Here every term is averaged over a diagnostic cell instead.
# Curl and Jacobian terms become edge integrals by Stokes' theorem,
#
#   ∬ curl(F) dA = ∮ F·t ds,   ∬ J(a, b) dA = ∮ a ∇b·t ds,
#
# and advection terms such as U·∇f use area quadrature, so no finite-element
# field is differentiated. Cell `(i, j)` spans the midpoints between
# neighboring diagnostic grid points.

using QuadGK: gauss

const CELL_EDGE_TERMS = 10
const CELL_AREA_TERMS = 2

"""
    CellBudgetQuadrature(; edge_panels=2, edge_points=8, area_points=3)

Quadrature for cell-averaged budgets: `edge_panels` composite Gauss--Legendre
panels of `edge_points` nodes along each cell edge, and an
`area_points × area_points` Gauss--Legendre rule over each cell.
"""
struct CellBudgetQuadrature
    edge_panels::Int
    edge_nodes::Vector{Float64}
    edge_weights::Vector{Float64}
    area_nodes::Vector{Float64}
    area_weights::Vector{Float64}
end

function CellBudgetQuadrature(; edge_panels=2, edge_points=8, area_points=3)
    edge_panels >= 1 || throw(ArgumentError("edge_panels must be positive"))
    edge_points >= 2 && area_points >= 1 ||
        throw(ArgumentError("edge_points must be ≥ 2 and area_points ≥ 1"))
    edge_nodes, edge_weights = gauss(edge_points)
    area_nodes, area_weights = gauss(area_points)
    return CellBudgetQuadrature(
        edge_panels, edge_nodes, edge_weights, area_nodes, area_weights,
    )
end

const DEFAULT_CELL_BUDGET_QUADRATURE = CellBudgetQuadrature()

water_point(problem, x, y) = problem.inside(x, y) && problem.depth(x, y) > 0

# Points that pass the polygon test can still fall just outside the meshed
# domain at the coast, where Gridap's cell search fails. Such cells are
# treated as dry.
function evaluate_if_located(f, args...)
    try
        return f(args...)
    catch error
        error isa Union{AssertionError,DomainError} && return nothing
        rethrow()
    end
end

"""
    cell_edge_integrands(solution, x, y, tx, ty)

Integrands along an edge with unit tangent `(tx, ty)`, in the order
`p_b ∇H·t`, `p_b ∇(H/f)·t`, `χ ∇(1/f)·t`, `χ ∇H·t / H²`, `τ_b·t`,
`τ_b·t / H`, `τ_b·t / f`, `τ·t`, `τ·t / H`, `τ·t / f`.
"""
function cell_edge_integrands(solution::PGSolution, x, y, tx, ty)
    problem = solution.problem
    H, f = problem.depth(x, y), problem.coriolis(x, y)
    H_t = problem.depth_x(x, y) * tx + problem.depth_y(x, y) * ty
    f_t = problem.coriolis_x(x, y) * tx + problem.coriolis_y(x, y) * ty
    pb = bottom_pressure(solution, x, y)
    χ = baroclinic_potential_energy(solution, x, y)
    bottom = bottom_stress(solution, x, y)
    surface = problem.wind_stress(x, y)
    τb_t = bottom.x * tx + bottom.y * ty
    τ_t = real(surface) * tx + imag(surface) * ty
    return (
        pb * H_t,
        pb * (H_t * f - H * f_t) / f^2,
        -χ * f_t / f^2,
        χ * H_t / H^2,
        τb_t,
        τb_t / H,
        τb_t / f,
        τ_t,
        τ_t / H,
        τ_t / f,
    )
end

"""
    cell_area_integrands(solution, x, y)

Area integrands `U·∇f` and `-U·∇(f/H)`.
"""
function cell_area_integrands(solution::PGSolution, x, y)
    problem = solution.problem
    H, f = problem.depth(x, y), problem.coriolis(x, y)
    H_x, H_y = problem.depth_x(x, y), problem.depth_y(x, y)
    f_x, f_y = problem.coriolis_x(x, y), problem.coriolis_y(x, y)
    current = depth_integrated_current(solution, x, y)
    q_x = (f_x * H - f * H_x) / H^2
    q_y = (f_y * H - f * H_y) / H^2
    return (
        current.U * f_x + current.V * f_y,
        -(current.U * q_x + current.V * q_y),
    )
end

"""
    bottom_stretching(solution, x, y)

`f w_b` with `w_b = -(u_b H_x + v_b H_y)` from the bottom kinematic condition.
It vanishes under the no-slip condition (14); the tests use it to check that
the velocity at the sea floor is zero.
"""
function bottom_stretching(solution::PGSolution, x, y)
    problem = solution.problem
    bottom_velocity = horizontal_velocity(solution, x, y, -problem.depth(x, y))
    return -problem.coriolis(x, y) * (
        bottom_velocity.u * problem.depth_x(x, y) +
        bottom_velocity.v * problem.depth_y(x, y)
    )
end

# Integral along the segment from `(x0, y0)` to `(x1, y1)` (axis-aligned) of
# the edge integrands, with the tangent pointing from start to end. Returns
# `nothing` if any quadrature point is dry.
function cell_edge_integral(solution, x0, y0, x1, y1, q::CellBudgetQuadrature)
    length_ = hypot(x1 - x0, y1 - y0)
    tx, ty = (x1 - x0) / length_, (y1 - y0) / length_
    total = zeros(CELL_EDGE_TERMS)
    for panel in 1:q.edge_panels
        s0 = (panel - 1) / q.edge_panels
        s1 = panel / q.edge_panels
        half = (s1 - s0) / 2
        for (node, weight) in zip(q.edge_nodes, q.edge_weights)
            s = s0 + half * (1 + node)
            x, y = x0 + s * (x1 - x0), y0 + s * (y1 - y0)
            water_point(solution.problem, x, y) || return nothing
            values = evaluate_if_located(cell_edge_integrands, solution, x, y, tx, ty)
            isnothing(values) && return nothing
            for k in 1:CELL_EDGE_TERMS
                total[k] += weight * half * length_ * values[k]
            end
        end
    end
    return total
end

function cell_area_integral(solution, xl, xr, yb, yt, q::CellBudgetQuadrature)
    total = zeros(CELL_AREA_TERMS)
    hx, hy = (xr - xl) / 2, (yt - yb) / 2
    for (nx, wx) in zip(q.area_nodes, q.area_weights),
        (ny, wy) in zip(q.area_nodes, q.area_weights)
        x, y = xl + hx * (1 + nx), yb + hy * (1 + ny)
        water_point(solution.problem, x, y) || return nothing
        values = evaluate_if_located(cell_area_integrands, solution, x, y)
        isnothing(values) && return nothing
        for k in 1:CELL_AREA_TERMS
            total[k] += wx * wy * hx * hy * values[k]
        end
    end
    return total
end

struct CellBudgetSamples
    circulation::Array{Float64,3}  # ∮ F·t ds / area, per edge integrand
    area::Array{Float64,3}         # U·∇f, -U·∇(f/H) cell means; f w_b at centre
    wet::BitMatrix
    timings::NamedTuple
end

# Avoid recomputing the shared edge and area integrals for each budget that
# uses the same solution and diagnostics.
const CELL_BUDGET_CACHE = Pair{Tuple{Any,Any,Any},CellBudgetSamples}[]
const CELL_BUDGET_CACHE_LOCK = ReentrantLock()

"""
    cell_budget_samples(solution, diagnostics; quadrature=CellBudgetQuadrature())

Cell-averaged circulation (Stokes) and area integrals shared by the BV, DIV,
GCVB, and TDVB budgets. Cell `(i, j)` spans the midpoints between neighboring
diagnostic points. It is used when its centered stencil is wet and every edge
and area quadrature point lies in water. Each interior edge is computed once.
"""
function cell_budget_samples(
    solution::PGSolution,
    diagnostics::PGDiagnostics;
    quadrature::CellBudgetQuadrature=DEFAULT_CELL_BUDGET_QUADRATURE,
)
    key = (solution, diagnostics, quadrature)
    lock(CELL_BUDGET_CACHE_LOCK) do
        for (cached_key, samples) in CELL_BUDGET_CACHE
            all(cached_key .=== key) && return samples
        end
        samples, timing = timed_stage("cell-averaged budget integrals") do
            compute_cell_budget_samples(solution, diagnostics, quadrature)
        end
        samples = CellBudgetSamples(
            samples.circulation, samples.area, samples.wet,
            (; integrals=timing),
        )
        push!(CELL_BUDGET_CACHE, key => samples)
        length(CELL_BUDGET_CACHE) > 4 && popfirst!(CELL_BUDGET_CACHE)
        return samples
    end
end

function compute_cell_budget_samples(solution, diagnostics, q)
    x, y = diagnostics.x, diagnostics.y
    nx, ny = length(x), length(y)
    xm = [(x[k] + x[k+1]) / 2 for k in 1:nx-1]
    ym = [(y[k] + y[k+1]) / 2 for k in 1:ny-1]
    candidate = [centered_stencil_is_wet(diagnostics.wet, i, j)
                 for i in 1:nx, j in 1:ny]

    # Horizontal edge (i, k) runs from (xm[i-1], ym[k]) to (xm[i], ym[k]);
    # vertical edge (k, j) runs from (xm[k], ym[j-1]) to (xm[k], ym[j]).
    horizontal = Matrix{Union{Nothing,Vector{Float64}}}(nothing, nx, ny - 1)
    vertical = Matrix{Union{Nothing,Vector{Float64}}}(nothing, nx - 1, ny)
    horizontal_needed = falses(nx, ny - 1)
    vertical_needed = falses(nx - 1, ny)
    for j in 2:ny-1, i in 2:nx-1
        candidate[i, j] || continue
        horizontal_needed[i, j-1] = horizontal_needed[i, j] = true
        vertical_needed[i-1, j] = vertical_needed[i, j] = true
    end
    horizontal_indices = Tuple.(findall(horizontal_needed))
    vertical_indices = Tuple.(findall(vertical_needed))
    Threads.@threads for index in eachindex(horizontal_indices)
        i, k = horizontal_indices[index]
        horizontal[i, k] =
            cell_edge_integral(solution, xm[i-1], ym[k], xm[i], ym[k], q)
    end
    Threads.@threads for index in eachindex(vertical_indices)
        k, j = vertical_indices[index]
        vertical[k, j] =
            cell_edge_integral(solution, xm[k], ym[j-1], xm[k], ym[j], q)
    end

    circulation = fill(NaN, nx, ny, CELL_EDGE_TERMS)
    area = fill(NaN, nx, ny, CELL_AREA_TERMS)
    wet_values = fill(false, nx, ny)
    cells = [(i, j) for j in 2:ny-1, i in 2:nx-1 if candidate[i, j]]
    Threads.@threads for index in eachindex(cells)
        i, j = cells[index]
        bottom, top = horizontal[i, j-1], horizontal[i, j]
        left, right = vertical[i-1, j], vertical[i, j]
        any(isnothing, (bottom, top, left, right)) && continue
        xl, xr, yb, yt = xm[i-1], xm[i], ym[j-1], ym[j]
        integrals = cell_area_integral(solution, xl, xr, yb, yt, q)
        isnothing(integrals) && continue
        cell_area = (xr - xl) * (yt - yb)
        # Counter-clockwise: bottom (+x), right (+y), top (-x), left (-y).
        circulation[i, j, :] .= (bottom .+ right .- top .- left) ./ cell_area
        area[i, j, :] .= integrals ./ cell_area
        wet_values[i, j] = true
    end
    return (; circulation, area, wet=BitMatrix(wet_values))
end

check_budget_method(method) = method in (:cell_average, :centered) ||
    throw(ArgumentError("method must be :cell_average or :centered; got $method"))

function masked_term(samples, values)
    out = fill(NaN, size(samples.wet))
    out[samples.wet] .= values[samples.wet]
    return out
end

function budget_timings(started, samples, residual, scale_terms)
    finite_l2(values) = sqrt(sum(abs2, values[samples.wet]))
    points = count(samples.wet)
    scale = max(maximum(finite_l2, scale_terms), eps())
    return (
        sampling=samples.timings.integrals,
        total_seconds=time() - started,
        relative_l2_residual=points > 0 ? finite_l2(residual) / scale : NaN,
    )
end

"""
    compute_barotropic_vorticity_diagnostics(solution, diagnostics;
        method=:cell_average, quadrature=CellBudgetQuadrature())

Maps of the signed terms in the barotropic vorticity equation (27) and its
closure residual. With `method=:cell_average` (the default), each term is its
average over the diagnostic cell around each grid point:
`J(p_b,H)` is `∮ p_b ∇H·t ds`, the stress curls are `Ek ∮ τ·t ds`, and
`U f_x + V f_y` uses area quadrature. The transport-divergence term takes its
exact fixed-lid value of zero (29). The residual therefore equals the
cell-averaged `-f ∇·U`, which is the finite-element mass-conservation error,
up to quadrature error. `method=:centered` gives the earlier centered
diagnostic-grid differences of sampled fields, which do not converge because
the finite-element pressure gradient is discontinuous.
"""
function compute_barotropic_vorticity_diagnostics(
    solution::PGSolution,
    diagnostics::PGDiagnostics;
    method=:cell_average,
    quadrature::CellBudgetQuadrature=DEFAULT_CELL_BUDGET_QUADRATURE,
)
    check_budget_method(method)
    method === :centered &&
        return centered_barotropic_vorticity_diagnostics(solution, diagnostics)
    started = time()
    samples = cell_budget_samples(solution, diagnostics; quadrature)
    c, a = samples.circulation, samples.area
    Ek = solution.problem.parameters.Ek
    divergence = masked_term(samples, zeros(size(samples.wet)))
    planetary = a[:, :, 1]
    pressure_torque = c[:, :, 1]
    bottom_curl = Ek .* c[:, :, 5]
    surface_curl = Ek .* c[:, :, 8]
    residual = divergence .+ planetary .- pressure_torque .- bottom_curl .-
        surface_curl
    timings = budget_timings(
        started, samples, residual,
        (divergence, planetary, pressure_torque, bottom_curl, surface_curl),
    )
    @info "BV diagnostic calculation completed" method seconds=round(timings.total_seconds; digits=3) relative_l2_residual=round(timings.relative_l2_residual; sigdigits=4)
    return PGBarotropicVorticityDiagnostics(
        diagnostics.x, diagnostics.y, divergence, planetary, pressure_torque,
        bottom_curl, surface_curl, residual, copy(samples.wet), timings,
    )
end

"""
    compute_vorticity_diagnostics(solution, diagnostics;
        method=:cell_average, quadrature=CellBudgetQuadrature())

The depth-integrated vorticity (DIV) balance (32). With the resolved no-slip
bottom boundary layer, `w_b = -(u_b H_x + v_b H_y) = 0`, and the viscous
vorticity flux through the sea floor equals `Ek ∇×τ_b + J(p_b,H)` (33), so the
DIV balance is identical to the barotropic vorticity balance (29). This is
therefore a thin alias for `compute_barotropic_vorticity_diagnostics`: the
planetary term, surface-stress curl, and residual are those of the BV balance,
the bottom term is `J(p_b,H) + Ek ∇×τ_b`, and the bottom stretching is zero.
"""
function compute_vorticity_diagnostics(
    solution::PGSolution,
    diagnostics::PGDiagnostics;
    method=:cell_average,
    quadrature::CellBudgetQuadrature=DEFAULT_CELL_BUDGET_QUADRATURE,
)
    bv = compute_barotropic_vorticity_diagnostics(solution, diagnostics; method, quadrature)
    stretching = fill(NaN, size(bv.wet))
    stretching[bv.wet] .= 0.0
    return PGVorticityDiagnostics(
        bv.x, bv.y, bv.planetary_vorticity_advection, stretching,
        bv.bottom_pressure_torque .+ bv.bottom_stress_curl,
        bv.surface_stress_curl, bv.residual, copy(bv.wet), bv.timings,
    )
end

"""
    compute_geostrophic_contour_vorticity_diagnostics(solution, diagnostics;
        method=:cell_average, quadrature=CellBudgetQuadrature())

Maps of the signed terms in the geostrophic-contour vorticity balance (34) and
its closure residual, as cell averages by default. `J(f/H,Ψ) = -U·∇(f/H)` uses
area quadrature with the analytical transport. JEBAR is
`J(1/H,χ) = ∮ χ ∇H·t / H² ds`, and the stress terms are `Ek ∮ (τ/H)·t ds`.
The residual equals the cell-averaged `(f/H) ∇·U`. `method=:centered` gives
the earlier centered-difference maps.
"""
function compute_geostrophic_contour_vorticity_diagnostics(
    solution::PGSolution,
    diagnostics::PGDiagnostics;
    method=:cell_average,
    quadrature::CellBudgetQuadrature=DEFAULT_CELL_BUDGET_QUADRATURE,
)
    check_budget_method(method)
    method === :centered && return centered_geostrophic_contour_vorticity_diagnostics(
        solution, diagnostics,
    )
    started = time()
    samples = cell_budget_samples(solution, diagnostics; quadrature)
    c, a = samples.circulation, samples.area
    Ek = solution.problem.parameters.Ek
    contour_advection = a[:, :, 2]
    jebar = c[:, :, 4]
    bottom_curl = Ek .* c[:, :, 6]
    surface_curl = Ek .* c[:, :, 9]
    residual = contour_advection .- jebar .+ bottom_curl .+ surface_curl
    timings = budget_timings(
        started, samples, residual,
        (contour_advection, jebar, bottom_curl, surface_curl),
    )
    @info "GCVB diagnostic calculation completed" method seconds=round(timings.total_seconds; digits=3) relative_l2_residual=round(timings.relative_l2_residual; sigdigits=4)
    return PGGeostrophicContourVorticityDiagnostics(
        diagnostics.x, diagnostics.y, contour_advection, jebar, bottom_curl,
        surface_curl, residual, copy(samples.wet), timings,
    )
end

"""
    compute_transport_divergence_vorticity_diagnostics(solution, diagnostics;
        method=:cell_average, quadrature=CellBudgetQuadrature())

Maps of the signed terms in the transport-divergence vorticity balance (37)
and its closure residual, as cell averages by default:
`J(p_b,H/f) = ∮ p_b ∇(H/f)·t ds`, `J(χ,1/f) = ∮ χ ∇(1/f)·t ds`, and the stress
terms are `Ek ∮ (τ/f)·t ds`. The residual equals the cell-averaged `-∇·U`.
`method=:centered` gives the earlier centered-difference maps.
"""
function compute_transport_divergence_vorticity_diagnostics(
    solution::PGSolution,
    diagnostics::PGDiagnostics;
    method=:cell_average,
    quadrature::CellBudgetQuadrature=DEFAULT_CELL_BUDGET_QUADRATURE,
)
    check_budget_method(method)
    method === :centered && return centered_transport_divergence_vorticity_diagnostics(
        solution, diagnostics,
    )
    started = time()
    samples = cell_budget_samples(solution, diagnostics; quadrature)
    c = samples.circulation
    Ek = solution.problem.parameters.Ek
    pressure_torque = c[:, :, 2]
    baroclinic_torque = c[:, :, 3]
    bottom_curl = Ek .* c[:, :, 7]
    surface_curl = Ek .* c[:, :, 10]
    residual = .-(pressure_torque .+ baroclinic_torque .+ bottom_curl .+
                  surface_curl)
    timings = budget_timings(
        started, samples, residual,
        (pressure_torque, baroclinic_torque, bottom_curl, surface_curl),
    )
    @info "TDVB diagnostic calculation completed" method seconds=round(timings.total_seconds; digits=3) relative_l2_residual=round(timings.relative_l2_residual; sigdigits=4)
    return PGTransportDivergenceVorticityDiagnostics(
        diagnostics.x, diagnostics.y, pressure_torque, baroclinic_torque,
        bottom_curl, surface_curl, residual, copy(samples.wet), timings,
    )
end
