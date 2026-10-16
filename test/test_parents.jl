using FourierContourCells: ChildList, BondList, integrate!, bond_energy, bond_gradient
using StaticArrays: SVector

# Bond term with AD gradient only: E = c |Δ|⁴ (+ type-dependent offset)
struct QuarticBond{T} <: BondTerm
    c::T
end
FourierContourCells.bond_energy(b::QuarticBond, Δ, ti, tj) = b.c * (Δ[1]^2 + Δ[2]^2)^2 + ti + tj

"Central finite-difference gradient of a bond energy."
function fd_bond_gradient(term, Δ, ti, tj; h = 1e-6)
    e(d) = bond_energy(term, d, ti, tj)
    return SVector((e(Δ + SVector(h, 0)) - e(Δ - SVector(h, 0))) / 2h,
        (e(Δ + SVector(0, h)) - e(Δ - SVector(0, h))) / 2h)
end

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

    @testset "bond gradients" begin
        Δ = SVector(0.3, -0.7)
        for term in (HarmonicBond(1.3), HarmonicBond(1.3; r0 = 0.4), QuarticBond(0.8))
            @test bond_gradient(term, Δ, 1, 2) ≈ fd_bond_gradient(term, Δ, 1, 2) rtol = 1e-7
        end
        @test bond_gradient(HarmonicBond(2.0), SVector(0.0, 0.0), 1, 1) == SVector(0.0, 0.0)
        @test bond_gradient(HarmonicBond(2.0; r0 = 0.5), SVector(0.0, 0.0), 1, 1) == SVector(0.0, 0.0)
        @test bond_energy(HarmonicBond(2.0; r0 = 0.5), SVector(0.3, 0.4), 1, 1) ≈ 0.0
    end

    @testset "bond lists" begin
        bl = BondList([(1, 2), (3, 1)], 3)
        @test Array(bl.count) == [2, 1, 1]
        @test sort(Array(bl.idx)[1:2, 1]) == [2, 3]
        @test FourierContourCells.nbonds(bl) == 2
        @test_throws ArgumentError BondList([(1, 1)], 3)
        @test_throws ArgumentError BondList([(1, 2), (2, 1)], 3)
        @test_throws ArgumentError BondList([(1, 4)], 3)
        coef = [1.0; zeros(10)] .* ones(1, 3)
        b = Bodies(zeros(3), zeros(3), zeros(3), coef, ones(3); parent = [1, 1, 2])
        @test parent_bonds(b) == [(2, 1), (3, 2)]
    end

    @testset "bond forces" begin
        # bonded pair across the periodic boundary; the third body is unbonded
        coef = [1.0; zeros(10)] .* ones(1, 3)
        bodies = Bodies([9.8, 0.4, 5.0], [3.0, 3.5, 5.0], zeros(3), coef, ones(3); type = [1, 2, 1])
        box = PeriodicBox(10.0, 10.0)
        Δ = [9.8 - 0.4 - 10, 3.0 - 3.5]                # minimum image r₁ - r₂
        for (terms, k) in (((HarmonicBond(0.7),), 0.7),
                (((OnTypes(HarmonicBond(0.7), 1, 2)),), 0.7),
                ((OnTypes(HarmonicBond(0.7), 1),), 0.0),     # type 2 excluded
                ((HarmonicBond(0.7), HarmonicBond(0.3)), 1.0))
            sim = Simulation(bodies, box; types = (BodyType(), BodyType()), energies = terms,
                bonds = [(1, 2)])
            compute_gradients!(sim)
            g = Array(sim.grad.gx), Array(sim.grad.gy)
            @test [g[1][1], g[2][1]] ≈ k .* Δ atol = 1e-14
            @test [g[1][2], g[2][2]] ≈ -k .* Δ atol = 1e-14
            @test [g[1][3], g[2][3]] == [0.0, 0.0]
        end
        @test_throws ArgumentError Simulation(bodies, box; types = (BodyType(), BodyType()),
            energies = (HarmonicBond(1.0), :notaterm))
    end

    @testset "bonded nucleus relaxes inside its cell" begin
        # parent drag: the bond is internal, so the cell stays put and the
        # offset follows explicit Euler of δ̇ = -μr k δ
        μ, k, δ0, n = 2.0, 0.5, 0.2, 300
        sim = cell_nucleus_sim(; nucleus = BodyType(μr = μ, drag = :parent))
        sim.bodies.x[2] += δ0
        sim = Simulation(sim.bodies, sim.box; types = sim.types, energies = (HarmonicBond(k),),
            bonds = parent_bonds(sim.bodies), dt = sim.dt)
        step!(sim, n)
        @test sim.bodies.x[1] == 5.0 && sim.bodies.y[1] == 5.0
        @test sim.bodies.x[2] - 5.0 ≈ δ0 * (1 - μ * k * sim.dt)^n rtol = 1e-12
    end

    @testset "prototyping hook" begin
        # constant extra force (1, -2) on body 3 via the hook
        calls = Ref(0)
        function extra_force!(sim)
            calls[] += 1
            sim.grad.gx[3] -= 1.0
            sim.grad.gy[3] += 2.0
        end
        coef = [1.0; zeros(10)] .* ones(1, 3)
        b = Bodies([5.0, 5.0, 1.0], [5.0, 5.0, 1.0], zeros(3), coef, ones(3); parent = [1, 1, 3],
            type = [1, 2, 1])
        sim = Simulation(b, PeriodicBox(10.0, 10.0); types = (BodyType(μr = 0.5), BodyType(drag = :parent)),
            dt = 0.01, hook! = extra_force!)
        step!(sim, 10)
        @test calls[] == 10
        @test unwrapped_positions(sim)[:, 3] ≈ [1.0, 1.0] .+ 0.5 * 0.1 .* [1.0, -2.0]
        @test unwrapped_positions(sim)[:, 1] ≈ [5.0, 5.0]
    end

    @testset "vcat and with_nuclei" begin
        coef = [1.0; zeros(10)] .* ones(1, 2)
        a = Bodies([1.0, 2.0], [1.0, 2.0], zeros(2), coef, ones(2); parent = [1, 1])
        b = Bodies([3.0, 4.0], [3.0, 4.0], zeros(2), coef, ones(2); type = 2, parent = [2, 2])
        c = vcat(a, b)
        @test nbodies(c) == 4
        @test c.parent == [1, 1, 4, 4]
        @test c.type == [1, 1, 2, 2]
        @test c.x == [1.0, 2.0, 3.0, 4.0]

        cells, box = hexagonal_monolayer(4, 4)
        bodies = with_nuclei(cells; ratio = 0.4)
        N = nbodies(cells)
        @test nbodies(bodies) == 2N
        @test bodies.parent == [1:N; 1:N]
        @test bodies.type == [fill(1, N); fill(2, N)]
        @test bodies.R0[(N + 1):end] ≈ 0.4 .* cells.R0
        @test bodies.x[(N + 1):end] == cells.x
        @test parent_bonds(bodies) == [(N + i, i) for i in 1:N]
    end

    @testset "observables by type and protrusion" begin
        cells, box = hexagonal_monolayer(4, 4)
        bodies = with_nuclei(cells; ratio = 0.5)
        N = nbodies(cells)
        bodies.x[N + 1] += 0.7           # nucleus 1 pokes 0.2 out of its (round) cell
        bodies.x[N + 2] += 0.2           # nucleus 2 lies 0.3 inside
        sim = Simulation(bodies, box; types = (BodyType(), BodyType(drag = :parent)), nθ = 64)
        @test areas(sim; type = 2) ≈ fill(π / 4, N)
        @test areas(sim; type = 1) ≈ fill(π, N)
        @test length(shape_index(sim; type = 2)) == N
        p = protrusion(sim)
        @test all(isnan, p[1:N])
        @test p[N + 1] ≈ 0.2 atol = 1e-12
        @test p[N + 2] ≈ -0.3 atol = 1e-12
        @test protrusion(sim; type = 2) == p[(N + 1):end]
    end
end
