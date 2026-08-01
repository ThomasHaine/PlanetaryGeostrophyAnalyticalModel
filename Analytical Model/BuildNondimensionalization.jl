using Revise
include("NonDimensionalizationTools.jl")
using .NonDimensionalizationTools

# Example physical scales
S = OceanScales(
    L  = 2.0e6,   # m
    H0 = 4.0e3,   # m
    f0 = 1.0e-4,  # 1/s
    N0 = 1.0e-3,  # 1/s
    ν0 = 1.0e-2,  # m^2/s
    κ0 = 1.0e-3,  # m^2/s
    γ  = 1.0e-5,  # 1/s
    τ0 = 0.1      # N/m^2
)

println("Derived scales:")
println("U0 = ", S.U0)
println("p0 = ", S.p0)
println("b0 = ", S.b0)
println("Ek = ", S.Ek)
println("Bu = ", S.Bu)
println("Pr = ", S.Pr)
println("Γ  = ", S.Γ)
println("δ  = ", S.δ)

# Example coordinates and fields
x  = [-1.0e6, 0.0, 1.0e6]
y  = [-5.0e5, 0.0, 5.0e5]
z  = [-4000.0, -2000.0, 0.0]

u  = [0.05, 0.06, 0.07]
v  = [0.02, 0.01, -0.01]
w  = [1e-5, 1e-5, 1e-5]
p  = [1.0e4, 2.0e4, 3.0e4]
b  = [1.0e-3, 2.0e-3, 3.0e-3]
τx = [0.05, 0.06, 0.07]
τy = [0.02, 0.03, 0.04]

x̂, ŷ, ẑ = nondimensionalize_coords(x, y, z, S)
û, v̂, ŵ, p̂, b̂, τ̂x, τ̂y = nondimensionalize_state(u, v, w, p, b, τx, τy, S)

println("\nNondimensional coordinates:")
println("x̂ = ", x̂)
println("ŷ = ", ŷ)
println("ẑ = ", ẑ)

println("\nNondimensional state:")
println("û = ", û)
println("v̂ = ", v̂)
println("ŵ = ", ŵ)
println("p̂ = ", p̂)
println("b̂ = ", b̂)
println("τ̂x = ", τ̂x)
println("τ̂y = ", τ̂y)

# Redimensionalize to verify round-trip
u2, v2, w2, p2, b2, τx2, τy2 = redimensionalize_state(û, v̂, ŵ, p̂, b̂, τ̂x, τ̂y, S)

println("\nRound-trip check for u:")
println(u2)
println("Original u:")
println(u)

# Bathymetry example
H = [0.0, 1500.0, 4000.0]
Ĥ = nondimensionalize_bathymetry(H, S)
H2 = redimensionalize_bathymetry(Ĥ, S)

println("\nBathymetry nondimensional:")
println(Ĥ)
println("Bathymetry redimensionalized:")
println(H2)