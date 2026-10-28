# Shared style for the manuscript figures drawn with Plots.jl (Figs. 3-10).
#
# Plots sizes a figure in pixels at 100 dpi and GR writes `size=(w, h)` as a
# w × h pt PDF, so a font of n pt is n × 100/72 pt in the PDF. The manuscript
# includes the PDF at a fixed width, so that font prints at
# n × (printed width in inches) × 100/w pt. `manuscript_style` returns font sizes that print at
# PRINTED_FONT_PT in every figure, whatever its pixel size. `slim_colorbar`
# draws a narrow colorbar as its own subplot, so that its width is set by the
# layout instead of by GR.
#
# Include after `using Plots`.

using Printf

# Printed text sizes in points, matching the text of Figs. 1 and 2.
const PRINTED_FONT_PT = (tick=6, guide=6.5, title=7, legend=5.5)

"""
    manuscript_style(width, printed_cm)

Font attributes for a figure `width` pixels wide that the manuscript includes
at `printed_cm`, so that its text prints at `PRINTED_FONT_PT`.
"""
function manuscript_style(width, printed_cm)
    scale = printed_cm / 2.54 * 100 / width
    pt(size) = round(Int, size / scale)
    return (;
        tickfontsize=pt(PRINTED_FONT_PT.tick),
        guidefontsize=pt(PRINTED_FONT_PT.guide),
        titlefontsize=pt(PRINTED_FONT_PT.title),
        legendfontsize=pt(PRINTED_FONT_PT.legend),
        framestyle=:box,
    )
end

"""
    slim_colorbar(clims, style; color)

A colorbar for `clims` and the color map `color`, drawn in its own subplot with
ticks on the right. A common power of ten is taken out of the tick labels and
shown above the bar; the panel titles name the field.
"""
function slim_colorbar(clims, style; color)
    lo, hi = clims
    exponent = floor(Int, log10(max(abs(lo), abs(hi))))
    factor = abs(exponent) >= 3 ? 10.0^exponent : 1.0
    ticks = Plots.PlotUtils.optimize_ticks(lo / factor, hi / factor; k_min=3, k_max=5)[1]
    ticks = filter(t -> lo / factor - 1e-9 <= t <= hi / factor + 1e-9, ticks)
    title = factor == 1 ? "" : "\$\\times 10^{$exponent}\$"
    values = collect(range(lo, hi; length=256))
    return heatmap([0.0, 1.0], values, repeat(values, 1, 2); color, clims,
        colorbar=false, legend=false, grid=false, xticks=false, ymirror=true,
        yticks=(ticks .* factor, [replace(@sprintf("%g", t), "-" => "−") for t in ticks]),
        xlims=(0, 1), ylims=clims, title, style..., titlefontsize=style.tickfontsize)
end

"""
    blank_panel()

An empty subplot, to keep the columns of a layout aligned.
"""
blank_panel() = plot(; framestyle=:none, grid=false, legend=false, xticks=false, yticks=false)
