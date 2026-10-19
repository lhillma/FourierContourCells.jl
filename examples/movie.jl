# Record an mp4 of a simulation.
#
#   julia -t auto --project=examples examples/movie.jl [backend=cpu|gpu] [T=Float32|Float64]
#
# Runs on a CUDA GPU in Float32 if one is available, else on the CPU in
# Float64 (see backend.jl). GLMakie renders off-screen while recording; use
# CairoMakie instead on a machine without OpenGL.

using FourierContourCells
using GLMakie
include("backend.jl")

backend, T = backend_and_precision()
bodies, box = hexagonal_monolayer(32, 32; packing = 0.95)
sim = Simulation(bodies, box;
    types = (BodyType(v0 = 0.06, Dr = 0.01, μab = 0.1),),
    energies = (Tension(0.01), Overlap(1.0)),
    dt = 0.005, backend, T)

# equilibrate without rendering
run!(sim, 20_000; every = 5_000, callback = s -> println("t = ", s.t))

mkpath(joinpath(@__DIR__, "..", "output"))
path = joinpath(@__DIR__, "..", "output", "soft_fluid.mp4")
FourierContourCells.record_movie(sim, path; frames = 600, steps_per_frame = 100, framerate = 30)
println("wrote ", path)
