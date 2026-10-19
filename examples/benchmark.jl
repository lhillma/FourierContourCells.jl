# Time steps of the soft-fluid benchmark (packing 0.95, Tension + Overlap) on
# the CPU and, if available, on a CUDA GPU, and print a kernel profile.
#
#   julia -t auto --project=examples examples/benchmark.jl [profile]
#
# With `profile`, CUDA.@profile shows per-kernel times of a few eager GPU steps.

using FourierContourCells
using CUDA

function soft_fluid(n; backend = CPU(), T = Float64, layout = FourierContourCells.default_layout(backend))
    bodies, box = hexagonal_monolayer(n, n; packing = 0.95)
    return Simulation(bodies, box; types = (BodyType(v0 = 0.06, Dr = 0.01, μab = 0.1),),
        energies = (Tension(0.01), Overlap(1.0)), dt = 0.005, backend, T, layout)
end

# `run!` uses CUDA graphs on the GPU (`graph = false`: eager `step!`)
function ms_per_step(sim; steps = 200, graph = true)
    sync() = sim.bodies.x isa CuArray && CUDA.synchronize()
    run!(sim, 20; graph)
    sync()
    t = @elapsed (run!(sim, steps; graph); sync())
    return round(t / steps * 1e3; digits = 3)
end

println("CPU, $(Threads.nthreads()) threads")
println("  N = 1024, Float64: ", ms_per_step(soft_fluid(32)), " ms/step")

if CUDA.functional()
    println("GPU: ", CUDA.name(CUDA.device()))
    for n in (16, 32, 128), T in (Float32, Float64)
        local sim = soft_fluid(n; backend = CUDABackend(), T)
        te, tg = ms_per_step(sim; graph = false), ms_per_step(sim)
        println("  N = $(n^2), $T: eager $te, graph $tg ms/step")
    end
    t = ms_per_step(soft_fluid(128; backend = CUDABackend(), T = Float32, layout = BodyParallel()))
    println("  N = 16384, Float32, BodyParallel: $t ms/step")
    if "profile" in ARGS
        sim = soft_fluid(128; backend = CUDABackend(), T = Float32)
        step!(sim, 20)
        display(CUDA.@profile step!(sim, 3))
    end
end
