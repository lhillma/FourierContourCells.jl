using FourierContourCells: ChildList, integrate!

"Cell (type 1) with nucleus (type 2, drag = :parent) at (5, 5), plus a free cell."
function cell_nucleus_sim(; nucleus = BodyType(drag = :parent), cell = BodyType(), dt = 0.01)
    coef = [1.0; zeros(10)] .* ones(1, 3)
    bodies = Bodies([5.0, 5.0, 1.0], [5.0, 5.0, 1.0], [0.3, -1.0, 2.0], coef, [1.0, 0.5, 1.0];
        type = [1, 2, 1], parent = [1, 1, 3])
    return Simulation(bodies, PeriodicBox(10.0, 10.0); types = (cell, nucleus), dt)
end

@testset "parents and children" begin
    @testset "ChildList" begin
        cl = ChildList(Int32[1, 1, 3, 1, 3, 6])
        @test Array(cl.count) == [2, 0, 1, 0, 0, 0]
        @test sort(Array(cl.idx)[1:2, 1]) == [2, 4]
        @test Array(cl.idx)[1, 3] == 5
        @test size(ChildList(Int32[1, 2]).idx) == (1, 2)
    end

    @testset "parent validation" begin
        @test_throws ArgumentError BodyType(drag = :cytoplasm)
        coef = [1.0; zeros(10)] .* ones(1, 2)
        box = PeriodicBox(10.0, 10.0)
        child = BodyType(drag = :parent)
        # parent drag without a parent
        b = Bodies([1.0, 2.0], [1.0, 2.0], zeros(2), coef, ones(2); type = [1, 2])
        @test_throws ArgumentError Simulation(b, box; types = (BodyType(), child))
        # parent with parent drag itself
        b = Bodies([1.0, 2.0], [1.0, 2.0], zeros(2), coef, ones(2); type = [2, 2], parent = [2, 1])
        @test_throws ArgumentError Simulation(b, box; types = (BodyType(), child))
        # parent id out of range
        b = Bodies([1.0, 2.0], [1.0, 2.0], zeros(2), coef, ones(2); parent = [1, 3])
        @test_throws ArgumentError Simulation(b, box; types = (BodyType(),))
    end

    @testset "displacements with parent drag" begin
        cell = BodyType(v0 = 0.3, Dt = 0.02, μr = 0.7)
        nucleus = BodyType(v0 = 0.1, Dt = 0.01, μr = 2.5, drag = :parent)
        sim = cell_nucleus_sim(; cell, nucleus)
        b, g = sim.bodies, sim.grad
        rng = Xoshiro(4)
        g.gx .= randn(rng, 3)
        g.gy .= randn(rng, 3)
        sim.noise .= randn(rng, 3, 3)
        θ, ξ, dt = copy(b.θ), copy(sim.noise), sim.dt
        x0, y0 = copy(b.x), copy(b.y)
        integrate!(b, g, sim.disp, sim.noise, sim.children, sim.types, dt, sim.box, sim.basis)
        e(i) = [cos(θ[i]), sin(θ[i])]
        G(i) = [g.gx[i], g.gy[i]]
        noise(i, D) = sqrt(2D * dt) .* ξ[1:2, i]
        # the cell bears its nucleus's force, the nucleus moves relative to it
        Δcell = dt .* (cell.v0 .* e(1) .- cell.μr .* (G(1) .+ G(2))) .+ noise(1, cell.Dt)
        Δnuc = Δcell .+ dt .* (nucleus.v0 .* e(2) .- nucleus.μr .* G(2)) .+ noise(2, nucleus.Dt)
        Δfree = dt .* (cell.v0 .* e(3) .- cell.μr .* G(3)) .+ noise(3, cell.Dt)
        @test [b.x[1] - x0[1], b.y[1] - y0[1]] ≈ Δcell
        @test [b.x[2] - x0[2], b.y[2] - y0[2]] ≈ Δnuc
        @test [b.x[3] - x0[3], b.y[3] - y0[3]] ≈ Δfree
    end

    @testset "nucleus is carried along" begin
        # passive nucleus, no forces: it follows the cell at v0 without lag
        sim = cell_nucleus_sim(; cell = BodyType(v0 = 0.5))
        run!(sim, 2000)   # 10 length units: crosses the periodic boundary
        X = unwrapped_positions(sim)
        @test X[:, 1] ≈ [5.0, 5.0] .+ 10 .* [cos(0.3), sin(0.3)]
        @test X[:, 2] ≈ X[:, 1]
    end
end
