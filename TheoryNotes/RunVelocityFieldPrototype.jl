# import Pkg
# Pkg.activate(@__DIR__)

include("VelocityFieldPrototype.jl")
using .VelocityFieldPrototype

println("Running manufactured solution test for VelocityFieldPrototype.jl...")
VelocityFieldPrototype.run_manufactured_test()