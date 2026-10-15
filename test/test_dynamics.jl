@testset "dynamics" begin
    @testset "hard area constraint" begin
        bodies, box = hexagonal_monolayer(2, 2)
        bodies.coef[2:end, :] .= 0.05 .* (1:10)
        sim = Simulation(bodies, box; types = (BodyType(),), nθ = 64)
        # shape bound enforced, area πR0² conserved
        c = Array(sim.bodies.coef)
        @test all(sum(hypot.(c[2:6, :], c[7:11, :]); dims = 1) .≤ sqrt(2 / 3) + 1e-12)
        @test areas(sim) ≈ fill(π, 4)
    end

    @testset "isolated cell self-propels" begin
        bodies = Bodies([1.0], [1.0], [0.3], [1.0; zeros(10);;], [1.0])
        sim = Simulation(bodies, PeriodicBox(5.0, 5.0); types = (BodyType(v0 = 0.5),),
            energies = (Tension(0.1), Overlap(1.0)), dt = 0.01)
        step!(sim, 3000)   # travels 15 units: crosses the periodic boundary
        @test unwrapped_positions(sim)[:, 1] ≈ [1.0, 1.0] .+ 15 .* [cos(0.3), sin(0.3)]
        @test 0 ≤ sim.bodies.x[1] < 5 && 0 ≤ sim.bodies.y[1] < 5
        @test sim.bodies.θ[1] ≈ 0.3
        @test maximum(abs, sim.bodies.coef[2:end, 1]) < 1e-14
        @test sim.t ≈ 30
    end

    @testset "rotational diffusion" begin
        N = 4000
        bodies = Bodies(rand(Xoshiro(1), N), rand(Xoshiro(2), N), zeros(N), [1.0; zeros(10)] .* ones(1, N), ones(N))
        sim = Simulation(bodies, PeriodicBox(1.0, 1.0); types = (BodyType(Dr = 0.1),), dt = 0.01, seed = 7)
        step!(sim, 500)
        @test var(Array(sim.bodies.θ); mean = 0) ≈ 2 * 0.1 * sim.t rtol = 0.1
    end
end
