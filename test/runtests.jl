using FourierContourCells
using LinearAlgebra
using Random
using Statistics
using Test

"Recompute derived quantities and gradients after modifying `sim.bodies` in place."
function refresh!(sim)
    FourierContourCells.enforce_shape!(sim.bodies, sim.basis)
    FourierContourCells.update_geometry!(sim.geom, sim.basis, sim.bodies)
    FourierContourCells.update_celllist!(sim.cells, sim.bodies)
    compute_gradients!(sim)
    return sim
end

"A deformed body (well inside the shape bound) as a coefficient column."
function deformed_coef(rng = Xoshiro(3); maxmode = 6, amplitude = 0.04)
    c = zeros(2maxmode - 1)
    c[2:end] .= amplitude .* (2 .* rand(rng, 2maxmode - 2) .- 1)
    c[1] = sqrt(1 - sum(abs2, c[2:end]) / 2)
    return c
end

@testset "FourierContourCells" begin
    include("test_geometry.jl")
end
