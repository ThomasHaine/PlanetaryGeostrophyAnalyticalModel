import Pkg
Pkg.activate(@__DIR__)

include("VelocityFieldPrototype.jl")
using .VelocityFieldPrototype

VelocityFieldPrototype.run_manufactured_test()
