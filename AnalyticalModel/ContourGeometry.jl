# Contour-geometry and grid-interpolation helpers shared by
# NostIsachsenDiagnostics.jl, GCVBDiagnostics.jl and
# IsobathCirculationDiagnostics.jl. Each module `include`s this file inside
# itself, so each keeps its own copy of the definitions and can still be
# loaded or omitted independently of the others. Requires `import Contour`.

signed_polygon_area(xs, ys) = 0.5sum(
    xs[index] * ys[index + 1] - xs[index + 1] * ys[index]
    for index in 1:(length(xs) - 1)
)

polygon_area(xs, ys) = abs(signed_polygon_area(xs, ys))

function contour_contains(x, y, xs, ys)
    inside = false
    previous = length(xs) - 1
    for current in 1:(length(xs) - 1)
        crosses = (ys[current] > y) != (ys[previous] > y)
        if crosses && x < (xs[previous] - xs[current]) *
                (y - ys[current]) / (ys[previous] - ys[current]) + xs[current]
            inside = !inside
        end
        previous = current
    end
    return inside
end

function interpolate_contour_function(q_value, contour_q, values)
    contour_q[1] <= q_value <= contour_q[end] || return NaN
    upper = searchsortedfirst(contour_q, q_value)
    upper == 1 && return values[1]
    lower = upper - 1
    weight = (q_value - contour_q[lower]) /
        (contour_q[upper] - contour_q[lower])
    return (1 - weight) * values[lower] + weight * values[upper]
end

"""
    bilinear_interpolate(x, y, field, xq, yq)

Bilinearly interpolate the real or complex `field` (sized `(length(x),
length(y))`, `field[i, j]` at `(x[i], y[j])`) at the point `(xq, yq)`.
Returns `NaN` (`NaN + 0im` for a complex field) if the query point falls
outside `[x[1], x[end]] × [y[1], y[end]]`, or if any of the four enclosing
grid corners is non-finite (e.g. masked out near the domain wall or a dry
cell), which excludes that contour segment from a line integral rather than
extrapolating across the mask.
"""
function bilinear_interpolate(x, y, field, xq, yq)
    missing_value = eltype(field) <: Complex ? NaN + 0.0im : NaN
    (x[1] <= xq <= x[end] && y[1] <= yq <= y[end]) || return missing_value
    i = clamp(searchsortedlast(x, xq), 1, length(x) - 1)
    j = clamp(searchsortedlast(y, yq), 1, length(y) - 1)
    tx = (xq - x[i]) / (x[i + 1] - x[i])
    ty = (yq - y[j]) / (y[j + 1] - y[j])
    c00, c10 = field[i, j], field[i + 1, j]
    c01, c11 = field[i, j + 1], field[i + 1, j + 1]
    all(isfinite, (c00, c10, c01, c11)) || return missing_value
    return (1 - tx) * (1 - ty) * c00 + tx * (1 - ty) * c10 +
        (1 - tx) * ty * c01 + tx * ty * c11
end

"""
    traced_closed_contours(x, y, field, wet; contour_levels, fill_value)

Trace closed contours of `field` on the rectilinear grid `x × y`. Non-finite
values are replaced by `fill_value` first. At each level, keep only the
largest-area closed loop (multiply-connected levels, e.g. separate per-basin
loops, are a known limitation) and orient it counter-clockwise so that
Green's- and Stokes'-theorem line integrals have a consistent sign. Returns a
vector of `(level, x, y)` named tuples sorted by level.

`field` must be sized `(length(x), length(y))` with `field[i, j]` at
`(x[i], y[j])`, i.e. *not* transposed, which is the layout
`Contour.contours` expects; passing the transpose silently swaps the axes of
every traced line.
"""
function traced_closed_contours(x, y, field, wet; contour_levels=20, fill_value)
    field_for_contours = copy(field)
    field_for_contours[.!isfinite.(field_for_contours)] .= fill_value
    contour_set = Contour.contours(x, y, field_for_contours, contour_levels)
    traced = NamedTuple{(:level, :x, :y),
        Tuple{Float64,Vector{Float64},Vector{Float64}}}[]
    for contour_level in Contour.levels(contour_set)
        closed_lines = Tuple{Vector{Float64},Vector{Float64}}[]
        for line in Contour.lines(contour_level)
            line_x, line_y = Contour.coordinates(line)
            hypot(line_x[end] - line_x[1], line_y[end] - line_y[1]) <
                1e-10 || continue
            push!(closed_lines, (collect(line_x), collect(line_y)))
        end
        isempty(closed_lines) && continue
        line_x, line_y = closed_lines[
            argmax(polygon_area(xs, ys) for (xs, ys) in closed_lines)
        ]
        if signed_polygon_area(line_x, line_y) < 0
            reverse!(line_x)
            reverse!(line_y)
        end
        push!(
            traced,
            (level=Contour.level(contour_level), x=line_x, y=line_y),
        )
    end
    return sort!(traced; by=t -> t.level)
end
