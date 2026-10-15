# Watch a simulation in real time (GLMakie window with run/pause and speed slider).
#
#   julia -t auto --project=examples examples/live.jl
#
# Parameters of the "soft fluid" phase of Saito & Ishihara (2024), Fig. 2C.

using FourierContourCells
using GLMakie

bodies, box = hexagonal_monolayer(32, 32; packing = 0.95)
sim = Simulation(bodies, box;
    types = (BodyType(v0 = 0.06, Dr = 0.01, μab = 0.1),),
    energies = (Tension(0.01), Overlap(1.0)),
    dt = 0.005)

live(sim; steps_per_frame = 50, color = :shape_index)
