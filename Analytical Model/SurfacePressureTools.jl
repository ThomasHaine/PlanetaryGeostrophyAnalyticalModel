using Gmsh
using Gridap
using GridapGmsh

"""
    generate_pressure_mesh(path, boundary; resolution)

Write a two-dimensional Gmsh mesh for the closed polygon `boundary`, whose
entries are `(x, y)` pairs. The mesh includes the physical line tag
`"boundary"` required by `solve_surface_pressure`.
"""
function generate_pressure_mesh(path, boundary; resolution, overwrite=false)
    if isfile(path)
        overwrite || return path
        rm(path)
    end
    gmsh.initialize()
    try
        gmsh.model.add("surface_pressure")
        occ = gmsh.model.occ
        point_tags = [
            occ.addPoint(x, y, 0.0, resolution) for (x, y) in boundary
        ]
        lines = [
            occ.addLine(point_tags[i], point_tags[mod1(i + 1, length(point_tags))])
            for i in eachindex(point_tags)
        ]
        loop = occ.addCurveLoop(lines)
        surface = occ.addPlaneSurface([loop])
        occ.synchronize()
        boundary_tag = gmsh.model.addPhysicalGroup(1, lines)
        gmsh.model.setPhysicalName(1, boundary_tag, "boundary")
        domain_tag = gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, domain_tag, "domain")
        gmsh.model.mesh.generate(2)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    return path
end

rectangle_boundary(Lx, Ly) = [
    (-Lx / 2, -Ly / 2), (Lx / 2, -Ly / 2),
    (Lx / 2, Ly / 2), (-Lx / 2, Ly / 2),
]

ellipse_boundary(ax, ay; points=96) = [
    (ax * cos(θ), ay * sin(θ)) for θ in range(0, 2π; length=points + 1)[1:end-1]
]

function pressure_coefficient(p, H, f)
    q = sqrt(im * f / (p.ν * p.Ek))
    return ((exp(2 * q * H) - 1) / (q * (1 + exp(2 * q * H))) - H) /
           (q^2 * p.Ek)
end

function wind_transport_coefficient(p, H, f, τ)
    q = sqrt(im * f / (p.ν * p.Ek))
    return (exp(q * H) - 1)^2 * τ / (q^2 * (1 + exp(2 * q * H)))
end

"""
    solve_surface_pressure(mesh_file, kernels, p, H, f, f_x, f_y, τ, B, bs, H_x, H_y)

Solve the same coupled Gridap surface-pressure problem used in
`PG_Solution_v0.1.ipynb`. The returned `gradps(x, y)` evaluates
`(∂x + i ∂y) p_s` for use in `velocity_b23_from_gradps`.
"""
function solve_surface_pressure(
    mesh_file, kernels, p, Hfun, ffun, f_x, f_y, τ, B, bs, H_x, H_y;
    order=2,
)
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2 * order)
    reffe = ReferenceFE(lagrangian, Float64, order)
    V = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")
    U = TrialFESpace(V)

    function A_fn(x)
        H = Hfun(x[1], x[2])
        return H > 0 ? pressure_coefficient(p, H, ffun(x[1], x[2])) : 1.0
    end
    function B_fn(x)
        H = Hfun(x[1], x[2])
        H > 0 || return 0.0 + 0.0im
        buoyancy = buoyancy_transport_b23(
            kernels, p, Hfun, ffun, f_x, f_y, B, bs, H_x, H_y, x[1], x[2],
        )
        return buoyancy + wind_transport_coefficient(
            p, H, ffun(x[1], x[2]), τ(x[1], x[2]),
        )
    end

    a(u, v) = ∫(real((∇(v) ⋅ VectorValue(1.0, -im)) *
                     (A_fn * (∇(u) ⋅ VectorValue(1.0, im)))) )dΩ
    l(v) = ∫(real((∇(v) ⋅ VectorValue(-1.0, im)) * B_fn))dΩ
    psurf = Gridap.solve(AffineFEOperator(a, l, U, V))
    ∇psurf = ∇(psurf)
    gradps(x, y) = begin
        gradient = evaluate(∇psurf, Point(x, y))
        gradient[1] + im * gradient[2]
    end
    return (; model, psurf, gradps, A_fn, B_fn)
end
