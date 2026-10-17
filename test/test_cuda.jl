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

        @testset "Float32 runs" begin
            sim = step!(gpu(; T = Float32), 50)
            @test eltype(sim.bodies.x) == Float32
            @test all(isfinite, Array(sim.bodies.coef))
        end
    end
else
    @info "No functional CUDA device: GPU tests skipped"
end
