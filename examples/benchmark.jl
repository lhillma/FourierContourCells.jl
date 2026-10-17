# Time steps of the soft-fluid benchmark (packing 0.95, Tension + Overlap) on
# the CPU and, if available, on a CUDA GPU, and print a kernel profile.
#
#   julia -t auto --project=examples examples/benchmark.jl [profile]
#
# With `profile`, CUDA.@profile shows per-kernel times of a few GPU steps.

using FourierContourCells
using CUDA

function soft_fluid(n; backend = CPU(), T = Float64, layout = FourierContourCells.default_layout(backend))
    bodies, box = hexagonal_monolayer(n, n; packing = 0.95)
    rng = backend isa CPU ? nothing : CUDA.RNG(1)
    return Simulation(bodies, box; types = (BodyType(v0 = 0.06, Dr = 0.01, μab = 0.1),),
        energies = (Tension(0.01), Overlap(1.0)), dt = 0.005, backend, T, rng, layout)
end

function ms_per_step(sim; steps = 200)
    sync() = sim.bodies.x isa CuArray && CUDA.synchronize()
    step!(sim, 20)
    sync()
    t = @elapsed (step!(sim, steps); sync())
    return round(t / steps * 1e3; digits = 3)
end

println("CPU, $(Threads.nthreads()) threads")
println("  N = 1024, Float64: ", ms_per_step(soft_fluid(32)), " ms/step")

if CUDA.functional()
    println("GPU: ", CUDA.name(CUDA.device()))
    for n in (32, 128), T in (Float32, Float64), layout in (BodyParallel(), NodeParallel())
        t = ms_per_step(soft_fluid(n; backend = CUDABackend(), T, layout))
        println("  N = $(n^2), $T, $(nameof(typeof(layout))): ", t, " ms/step")
    end
    if "profile" in ARGS
        sim = soft_fluid(128; backend = CUDABackend(), T = Float32)
        step!(sim, 20)
        display(CUDA.@profile step!(sim, 3))
    end
end
