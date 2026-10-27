# Helpers shared by the classical-limit comparisons of manuscript Section 6c
# (StommelComparison.jl, BetaPlumeComparison.jl, BoundaryBetaPlumeComparison.jl,
# TopographicSteeringComparison.jl) and by VerificationConvergence.jl.
# Include after `using PlanetaryGeostrophy`.

relative_l2(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

"""
    gaussian_curl_stress(centre, radius, amplitude)

Return `(tau, wind_curl)`: an azimuthal stress `tau(x, y) = τx + iτy` whose
curl is exactly the Gaussian `wind_curl(x, y)` of the given amplitude and
radius about `centre`.
"""
function gaussian_curl_stress(centre, radius, amplitude)
    radius_squared(x, y) = (x - centre[1])^2 + (y - centre[2])^2
    function stress_factor(x, y)
        r2 = radius_squared(x, y)
        r2 < 1e-14 && return amplitude / 2
        return amplitude * radius^2 * (-expm1(-r2 / (2radius^2))) / r2
    end
    tau(x, y) = complex(-stress_factor(x, y) * (y - centre[2]),
        stress_factor(x, y) * (x - centre[1]))
    wind_curl(x, y) = amplitude * exp(-radius_squared(x, y) / (2radius^2))
    return tau, wind_curl
end

"""
    pg_streamfunction(solution, x, y; inside=(x, y) -> true)

PG streamfunction from the transport, `Ψ(x, y) = -∫_x^{east} V dx'`, integrated
westward by the trapezoidal rule from the eastern end of each row of the grid.
Each row stops at the first point outside `inside`; such points are NaN.
"""
function pg_streamfunction(solution, x, y; inside=(_, _) -> true)
    Ψ = fill(NaN, length(x), length(y))
    Threads.@threads for j in eachindex(y)
        inside(x[end], y[j]) || continue
        Ψ[end, j] = 0.0
        V_east = depth_integrated_current(solution, x[end], y[j]).V
        for i in length(x)-1:-1:1
            inside(x[i], y[j]) || break
            V = depth_integrated_current(solution, x[i], y[j]).V
            Ψ[i, j] = Ψ[i+1, j] - 0.5 * (V + V_east) * (x[i+1] - x[i])
            V_east = V
        end
    end
    return Ψ
end

# Bilinear interpolation of a field on the (sorted) grid `x` × `y`.
function interpolate_grid(x, y, F, px, py)
    i = clamp(searchsortedlast(x, px), 1, length(x) - 1)
    j = clamp(searchsortedlast(y, py), 1, length(y) - 1)
    a, b = (px - x[i]) / (x[i+1] - x[i]), (py - y[j]) / (y[j+1] - y[j])
    return (1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i+1, j] +
        (1 - a) * b * F[i, j+1] + a * b * F[i+1, j+1]
end
