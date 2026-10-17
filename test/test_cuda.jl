# GPU paths (node-parallel kernels, stacked synthesis, Philox on the device).
# Skipped without a functional CUDA device.
using CUDA

if CUDA.functional()
    @testset "CUDA" begin
        backend = CUDA.CUDABackend()
        gpu(layout = NodeParallel(); kw...) = layout_sim(layout; backend, kw...)

        @testset "gradients match the CPU ($(nameof(typeof(l))), nθ = $nθ)" for
                l in (NodeParallel(), BodyParallel()), nθ in (40, 37)
            a = refresh!(layout_sim(BodyParallel(); nθ))
            b = refresh!(gpu(l; nθ))
            @test FourierContourCells.default_layout(backend) == NodeParallel()
            @test Array(b.geom.R) ≈ Array(a.geom.R)
            @test Array(b.geom.ddR) ≈ Array(a.geom.ddR)
            for f in (:GR, :GD, :GDD, :P, :gx, :gy, :gθ, :gcoef, :Ebody)
                @test Array(getfield(b.grad, f)) ≈ getfield(a.grad, f) rtol = 1e-10 atol = 1e-10
            end
        end

        @testset "same noise as the CPU" begin
            key = FourierContourCells.philox_key(9)
            for T in (Float32, Float64)
                c = FourierContourCells.philox_noise!(zeros(T, 3, 1000), key, 12)
                g = FourierContourCells.philox_noise!(CUDA.zeros(T, 3, 1000), key, 12)
                @test Array(g) ≈ c rtol = 4eps(T)
            end
        end

        @testset "short trajectory matches the CPU (Float64)" begin
            a, b = layout_sim(BodyParallel()), gpu()
            step!(a, 30)
            step!(b, 30)
            @test Array(b.bodies.x) ≈ a.bodies.x rtol = 1e-8
            @test Array(b.bodies.θ) ≈ a.bodies.θ rtol = 1e-8
            @test Array(b.bodies.coef) ≈ a.bodies.coef rtol = 1e-8
            @test b.neighbors.builds == a.neighbors.builds
        end

        @testset "graph mode" begin
            state(s) = map(Array, (s.bodies.x, s.bodies.y, s.bodies.θ, s.bodies.coef, s.bodies.imgx,
                s.grad.gcoef, s.grad.gx, s.disp))
            same(a, b) = state(a) == state(b) && a.t == b.t && a.step == b.step &&
                a.neighbors.builds == b.neighbors.builds
            # fast cells and a thin skin: many rebuilds, frozen chunks
            fast(; kw...) = gpu(; v0 = 2.0, skin = 0.05, kw...)

            a, b = fast(), fast()
            step!(a, 200)
            run!(b, 200)
            @test a.neighbors.builds > 10
            @test same(a, b)
            @test b.graph[] !== nothing

            # callbacks between graph chunks
            a, b = fast(), fast()
            seen = Int[]
            step!(a, 100)
            run!(b, 100; every = 7, callback = s -> push!(seen, s.step))
            @test seen == [7:7:98; 100]
            @test same(a, b)

            # a new time step and a reallocated neighbour list force a re-capture
            g = b.graph[]
            a.dt = b.dt = 0.004
            step!(a, 30)
            run!(b, 30)
            @test b.graph[] !== g
            @test same(a, b)
            for s in (a, b)
                s.neighbors.idx = CUDA.zeros(Int32, 1, nbodies(s))
                s.neighbors.valid = false
            end
            g = b.graph[]
            step!(a, 30)
            run!(b, 30)
            @test b.graph[] !== g && size(b.neighbors.idx, 1) > 1
            @test same(a, b)

            # eager fallbacks: graph = false, a hook, no skin; body terms only
            c = fast()
            run!(c, 20; graph = false)
            @test c.graph[] === nothing
            h = fast(; hook! = s -> nothing)
            @test !FourierContourCells.graph_capable(h)
            @test same(step!(fast(; hook! = s -> nothing), 20), run!(h, 20))
            @test !FourierContourCells.graph_capable(gpu(; skin = 0))
            bodies, box = hexagonal_monolayer(4, 4)
            only_body(; kw...) = Simulation(bodies, box; types = (BodyType(v0 = 0.3, Dr = 0.5),),
                energies = (PerimeterElasticity(0.2, 3.7),), backend, kw...)
            @test same(step!(only_body(), 50), run!(only_body(), 50))
        end

        @testset "Float32 runs" begin
            sim = step!(gpu(; T = Float32), 50)
            @test eltype(sim.bodies.x) == Float32
            @test all(isfinite, Array(sim.bodies.coef))
        end
    end
else
    @info "No functional CUDA device: GPU tests skipped"
end
