"Parse the reference fixture written by the original C++ code."
function read_reference(path)
    init = Vector{Vector{Float64}}()
    snaps = Dict{Int, Vector{Vector{Float64}}}()
    current = -1
    for l in eachline(path)
        if startswith(l, "#init")
            push!(init, parse.(Float64, split(l)[2:end]))
        elseif startswith(l, "#T=")
            current = parse(Int, split(l[4:end])[1]) + 1   # state after T+1 steps
            snaps[current] = Vector{Vector{Float64}}()
        elseif startswith(l, "#cell")
            push!(snaps[current], parse.(Float64, split(l)[2:end]))
        end
    end
    return reduce(hcat, init), Dict(k => reduce(hcat, v) for (k, v) in snaps)
end

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

    @testset "regression against reference implementation" begin
        init, snaps = read_reference(joinpath(@__DIR__, "data", "reference_N64.txt"))
        N = size(init, 2)
        Ls = sqrt(N * π / (0.95 * 2sqrt(3)))
        coef = [1.0; zeros(10)] .* ones(1, N)
        bodies = Bodies(init[1, :], init[2, :], init[3, :], coef, ones(N))
        sim = Simulation(bodies, PeriodicBox(2Ls, sqrt(3) * Ls);
            types = (BodyType(v0 = 0.06, Dr = 0.0, μab = 0.1),),
            energies = (Tension(0.01), Overlap(1.0)), dt = 0.005)
        for n in sort(collect(keys(snaps)))
            step!(sim, n - sim.step)
            ref = snaps[n]
            pos = unwrapped_positions(sim)
            @test pos[1, :] ≈ ref[4, :] atol = 1e-10
            @test pos[2, :] ≈ ref[5, :] atol = 1e-10
            @test Array(sim.bodies.θ) ≈ ref[3, :] atol = 1e-10
            @test Array(sim.bodies.coef)[2:end, :] ≈ ref[6:15, :] atol = 1e-10
        end
    end
end
