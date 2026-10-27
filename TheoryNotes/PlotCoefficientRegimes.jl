# Regime figure for the column coefficients 𝔄 (60) and S (57) (labelled A and S in the plot).
#
# With uniform viscosity, f𝔄/δ_E, Ek S/δ_E, and the surface-stress weight in T
# depend only on s = H/δ_E, where δ_E = sqrt(2νEk/|f|) is the Ekman depth. The
# figure evaluates the package's own coefficient functions at two Ekman numbers
# (to show this collapse) and overlays the deep (s ≫ 1) and shallow (s ≪ 1)
# asymptotes.
#
# Run from the repository root:
#   julia --project=PlanetaryGeostrophy TheoryNotes/PlotCoefficientRegimes.jl
# Writes TheoryNotes/coefficient_regimes.{pdf,png}. Strings delimited by $...$
# are rendered as LaTeX by the GR backend.

ENV["GKSwstype"] = "100"
using PlanetaryGeostrophy
using Plots

const PG = PlanetaryGeostrophy

ekman_depth(p, f) = sqrt(2 * p.ν * p.Ek / abs(f))

function scaled_coefficients(Ek, s_values; f=1.0, ν=1.0)
    p = PG.VelocityParameters(Ek, 1.0, ν, 1.0, f, 0.0)
    δ = ekman_depth(p, f)
    q = sqrt(im * f / (ν * Ek))
    A = [PG.pressure_coefficient(p, s * δ, f) for s in s_values]
    S = [PG.bottom_stress_coefficient(p, s * δ, f) for s in s_values]
    # Weight of the surface stress in the bottom-stress forcing T (58):
    # -2e^{qH}/(1 + e^{2qH}) = -sech(qH).
    wind = [abs(2exp(q * s * δ) / (1 + exp(2q * s * δ))) for s in s_values]
    return (;
        steering=f .* imag.(A) ./ δ,
        diffusivity=-f .* real.(A) ./ δ,
        stress_magnitude=Ek .* abs.(S) ./ δ,
        stress_angle=rad2deg.(angle.(S)),
        wind,
    )
end

s = 10 .^ range(-1.3, log10(20.0); length=400)
s_markers = 10 .^ range(-1.2, log10(18.0); length=17)
reference = scaled_coefficients(1e-2, s)
check = scaled_coefficients(1e-3, s_markers)

deep = s .> 0.7
shallow = s .< 1.5
regime_lines(panel) = vline!(panel, [1.0, 3.0]; color=:gray, linestyle=:dot, label=false)
common = (; xscale=:log10, xlabel="\$s = H/\\delta_E\$", legend=:topleft, framestyle=:box,
    minorgrid=false, titlefontsize=10, guidefontsize=9, legendfontsize=7)

steering = plot(s, reference.steering; yscale=:log10, lw=2, color=:black,
    label="exact", title="(a) steering \$f\\,A_i/\\delta_E\$", ylabel="\$f\\,A_i/\\delta_E\$", common..., legend=:bottomright)
scatter!(steering, s_markers, check.steering; color=:black, ms=3, label="exact, Ek/10")
plot!(steering, s, s; color=:gray, ls=:dot, label="inviscid \$H/f\$")
plot!(steering, s[deep], s[deep] .- 0.5; color=:steelblue, ls=:dash, lw=1.5,
    label="deep: \$(H-\\delta_E/2)/f\$")
plot!(steering, s[shallow], 8 / 15 .* s[shallow] .^ 5; color=:firebrick, ls=:dashdot,
    lw=1.5, label="shallow: \$(8/15)\\,s^5\$")
regime_lines(steering)

diffusivity = plot(s, reference.diffusivity; yscale=:log10, lw=2, color=:black,
    label="exact", title="(b) friction \$-f\\,A_r/\\delta_E\$", ylabel="\$-f\\,A_r/\\delta_E\$", common..., legend=:bottomright)
scatter!(diffusivity, s_markers, check.diffusivity; color=:black, ms=3, label="exact, Ek/10")
hline!(diffusivity, [0.5]; color=:steelblue, ls=:dash, lw=1.5, label="deep: \$\\delta_E/(2f)\$")
plot!(diffusivity, s[shallow], 2 / 3 .* s[shallow] .^ 3; color=:firebrick, ls=:dashdot,
    lw=1.5, label="shallow: \$(2/3)\\,s^3\$")
regime_lines(diffusivity)

ratio = plot(s, reference.steering ./ reference.diffusivity; yscale=:log10, lw=2,
    color=:black, label="exact", title="(c) steering/friction \$A_i/(-A_r)\$",
    ylabel="\$A_i/(-A_r)\$", common..., legend=:bottomright)
scatter!(ratio, s_markers, check.steering ./ check.diffusivity; color=:black, ms=3,
    label="exact, Ek/10")
plot!(ratio, s[deep], 2 .* s[deep] .- 1; color=:steelblue, ls=:dash, lw=1.5,
    label="deep: \$2s-1\$")
plot!(ratio, s[shallow], 4 / 5 .* s[shallow] .^ 2; color=:firebrick, ls=:dashdot,
    lw=1.5, label="shallow: \$(4/5)\\,s^2\$")
hline!(ratio, [1.0]; color=:gray, ls=:dot, label=false)
regime_lines(ratio)

stress = plot(s, reference.stress_magnitude; yscale=:log10, lw=2, color=:black,
    label="exact", title="(d) bottom stress \$\\mathrm{Ek}\\,|S|/\\delta_E\$",
    ylabel="\$\\mathrm{Ek}\\,|S|/\\delta_E\$", common...)
scatter!(stress, s_markers, check.stress_magnitude; color=:black, ms=3, label="exact, Ek/10")
hline!(stress, [1 / sqrt(2)]; color=:steelblue, ls=:dash, lw=1.5, label="deep: \$1/\\sqrt{2}\$")
plot!(stress, s[shallow], s[shallow]; color=:firebrick, ls=:dashdot, lw=1.5,
    label="shallow: \$s\$")
regime_lines(stress)

angle_panel = plot(s, reference.stress_angle; lw=2, color=:black, label="exact",
    title="(e) bottom-stress angle \$\\arg S\$", ylabel="degrees",
    ylims=(-60, 10), common..., legend=:bottomleft)
scatter!(angle_panel, s_markers, check.stress_angle; color=:black, ms=3, label="exact, Ek/10")
hline!(angle_panel, [-45.0]; color=:steelblue, ls=:dash, lw=1.5, label="deep: \$-45^\\circ\$")
hline!(angle_panel, [0.0]; color=:firebrick, ls=:dashdot, lw=1.5, label="shallow: \$0^\\circ\$")
regime_lines(angle_panel)

wind = plot(s, reference.wind; yscale=:log10, lw=2, color=:black, label="exact",
    title="(f) wind stress at the bed \$|\\mathrm{sech}\\,\\varphi H|\$",
    ylabel="fraction of surface stress", ylims=(1e-8, 2), common..., legend=:bottomleft)
scatter!(wind, s_markers, check.wind; color=:black, ms=3, label="exact, Ek/10")
plot!(wind, s[deep], 2 .* exp.(-s[deep]); color=:steelblue, ls=:dash, lw=1.5,
    label="deep: \$2e^{-s}\$")
regime_lines(wind)

figure = plot(steering, diffusivity, ratio, stress, angle_panel, wind;
    layout=(2, 3), size=(1150, 680), left_margin=4Plots.mm, bottom_margin=4Plots.mm)
for extension in ("pdf", "png")
    savefig(figure, joinpath(@__DIR__, "coefficient_regimes.$extension"))
end

# Accuracy of the deep asymptotes, used in the manuscript text.
for s_check in (1.0, 2.0, 3.0, 5.0)
    c = scaled_coefficients(1e-2, [s_check])
    println(
        "s = $s_check: steering error ", round(abs(c.steering[1] - (s_check - 0.5)) / c.steering[1]; sigdigits=2),
        ", friction error ", round(abs(c.diffusivity[1] - 0.5) / 0.5; sigdigits=2),
        ", |S| error ", round(abs(c.stress_magnitude[1] - 1 / sqrt(2)) * sqrt(2); sigdigits=2),
        ", wind at bed ", round(c.wind[1]; sigdigits=2),
    )
end
