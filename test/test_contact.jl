using FourierContourCells: contour_radius, minimum_image, node_angle, _pressure

"Brute-force contact pressure over all pairs (host reference implementation)."
function brute_force_pressure(sim)
    b = FourierContourCells.to_host(sim.bodies)
    R = Array(sim.geom.R)
    lab = Array(sim.geom.lab)
    K = nmodes(sim.basis)
    P = zeros(size(R))
    for i in 1:nbodies(sim), k in 1:(sim.basis.nθ)
        φ = node_angle(sim.basis, k)
        ux, uy = b.x[i] + R[k, i] * cos(φ), b.y[i] + R[k, i] * sin(φ)
        for j in 1:nbodies(sim)
            j == i && continue
            dx = minimum_image(ux - b.x[j], sim.box.Lx)
            dy = minimum_image(uy - b.y[j], sim.box.Ly)
            d = contour_radius(lab, j, b.R0[j], K, atan(dy, dx)) - hypot(dx, dy)
            P[k, i] += _pressure(sim.contact_terms, d, b.type[i], b.type[j], b.parent[i] == b.parent[j])
        end
    end
    return P
end

"Two-body simulation: body 1 at the origin, body 2 at (d, 0)."
function two_body_sim(terms; d = 1.5, R0 = (1.0, 1.0), nθ = 40, deformed = false, θ = (0.3, -1.1),
        type = (1, 1), parent = (1, 2), ntypes = 1)
    coef = [1.0; zeros(10)] .* ones(1, 2)
    deformed && (coef[:, 1] = deformed_coef(Xoshiro(5)); coef[:, 2] = deformed_coef(Xoshiro(6)))
    bodies = Bodies([5.0, 5.0 + d], [5.0, 5.0], collect(θ), coef, collect(R0);
        type = collect(type), parent = collect(parent))
    sim = Simulation(bodies, PeriodicBox(10.0, 10.0); types = ntuple(_ -> BodyType(), ntypes),
        energies = terms, nθ)
    refresh!(sim)
    return sim
end

"""
Energy of body 1 in the field of body 2, E₁ = Σₖ Δφ ∫₀^{Rₖ} g(d₂) s ds, by radial
midpoint quadrature (the angular discretisation matches the model's).
"""
function field_energy(sim; nr = 4000)
    b = FourierContourCells.to_host(sim.bodies)
    R = Array(sim.geom.R)
    lab = Array(sim.geom.lab)
    K = nmodes(sim.basis)
    E = 0.0
    for k in 1:(sim.basis.nθ)
        φ = node_angle(sim.basis, k)
        for t in ((1:nr) .- 0.5) ./ nr
            s = R[k, 1] * t
            dx, dy = b.x[1] + s * cos(φ) - b.x[2], b.y[1] + s * sin(φ) - b.y[2]
            d = contour_radius(lab, 2, b.R0[2], K, atan(dy, dx)) - hypot(dx, dy)
            E += _pressure(sim.contact_terms, d, b.type[1], b.type[2], false) * s * R[k, 1] / nr
        end
    end
    return E * sim.basis.Δφ
end

@testset "contact" begin
    @testset "Overlap couplings" begin
        o = Overlap([1.0 2.0; 2.0 3.0]; related = [1.0 -0.5; -0.5 1.0])
        @test pressure(o, 0.1, 1, 2, false) == 2.0
        @test pressure(o, 0.1, 1, 2, true) == -0.5
        @test pressure(o, -0.1, 2, 2, false) == 0.0
        @test pressure(Overlap(1.0), 0.0, 1, 1, false) == 1.0
        @test pressure(LinearRepulsion(2.0), 0.25, 1, 1, false) == 0.5
    end

    @testset "cell list matches brute force" begin
        bodies, box = hexagonal_monolayer(8, 8; packing = 0.98, rng = Xoshiro(2))
        bodies.x .+= 0.4 .* randn(Xoshiro(3), 64)
        bodies.y .+= 0.4 .* randn(Xoshiro(4), 64)
        bodies.x .= mod.(bodies.x, box.Lx)
        bodies.y .= mod.(bodies.y, box.Ly)
        bodies.coef[2:end, :] .= 0.05 .* randn(Xoshiro(5), 10, 64)
        sim = Simulation(bodies, box; types = (BodyType(),), energies = (Overlap(1.0), LinearRepulsion(0.3)))
        refresh!(sim)
        @test sim.cells.ncx ≥ 3 && sim.cells.ncy ≥ 3
        @test count(>(0), sim.grad.P) > 100
        @test Array(sim.grad.P) ≈ brute_force_pressure(sim)
        # neighbour list capacity grows when exceeded
        sim.neighbors.idx = zeros(Int32, 2, nbodies(sim))
        refresh!(sim)
        @test size(sim.neighbors.idx, 1) ≥ maximum(sim.neighbors.count) > 2
        @test Array(sim.grad.P) ≈ brute_force_pressure(sim)
    end

    @testset "two discs: overlap force vs exact" begin
        # overlap area of unit discs at distance d: dA/dd = -√(4 - d²); the
        # Heaviside integrand makes the angular quadrature first order in Δφ
        d = 1.5
        sim = two_body_sim((Overlap(1.0),); d, nθ = 8000)
        @test sim.grad.gx[1] ≈ sqrt(4 - d^2) rtol = 1e-3
        @test sim.grad.gx[2] ≈ -sqrt(4 - d^2) rtol = 1e-3
        @test abs(sim.grad.gy[1]) < 1e-6
    end

    @testset "finite-difference gradient (LinearRepulsion)" begin
        sim = two_body_sim((LinearRepulsion(1.0),); d = 1.6, deformed = true, nθ = 200)
        g = (x = sim.grad.gx[1], θ = sim.grad.gθ[1], coef = Array(sim.grad.gcoef)[2:end, 1])
        coef0 = Array(sim.bodies.coef)
        function E_at(; δx = 0.0, δθ = 0.0, δc = zeros(10))
            c = copy(coef0)
            c[2:end, 1] .+= δc
            copyto!(sim.bodies.coef, c)
            sim.bodies.x[1] = 5.0 + δx
            sim.bodies.θ[1] = 0.3 + δθ
            refresh!(sim)
            return field_energy(sim)
        end
        h = 1e-5
        @test (E_at(δx = h) - E_at(δx = -h)) / 2h ≈ g.x rtol = 2e-3
        @test (E_at(δθ = h) - E_at(δθ = -h)) / 2h ≈ g.θ rtol = 1e-4
        for r in (1, 4, 8)
            e = zeros(10)
            e[r] = h
            # tolerance limited by the radial quadrature of the reference energy
            @test (E_at(δc = e) - E_at(δc = -e)) / 2h ≈ g.coef[r] rtol = 1e-3 atol = 1e-8
        end
    end

    @testset "nucleus attraction to own cell" begin
        # type 1 = cell (R0 = 1), type 2 = nucleus (R0 = 0.4), same parent
        J = [1.0 1.0; 1.0 1.0]
        Jrel = [0.0 -2.0; -2.0 0.0]
        terms = (Overlap(J; related = Jrel),)
        partial = two_body_sim(terms; d = 0.9, R0 = (1.0, 0.4), type = (1, 2), parent = (1, 1), ntypes = 2, nθ = 400)
        @test partial.grad.gx[2] > 0.1   # force -∂E/∂x pulls the nucleus back inside
        @test partial.grad.gx[1] < -0.1  # and the cell towards the nucleus
        inside = two_body_sim(terms; d = 0.3, R0 = (1.0, 0.4), type = (1, 2), parent = (1, 1), ntypes = 2, nθ = 400)
        @test abs(inside.grad.gx[2]) < 1e-10
        @test abs(inside.grad.gx[1]) < 1e-10
    end

    @testset "Verlet skin" begin
        function skin_sim(skin; v0 = 1.0)
            bodies, box = hexagonal_monolayer(6, 6; packing = 0.97, rng = Xoshiro(2))
            Simulation(bodies, box; types = (BodyType(; v0, Dr = 1.0, Dt = 0.01, μab = 1.0),),
                energies = (Tension(0.1), Overlap(1.0), LinearRepulsion(0.3)), dt = 0.01, skin, seed = 3)
        end
        @test_throws ArgumentError skin_sim(-0.1)
        # the list built with the skin is a superset of the interacting pairs at every step
        # (P from step! belongs to the state before the step, so recompute it)
        sim = skin_sim(0.2)
        exact = true
        for _ in 1:60
            step!(sim)
            FourierContourCells.update_neighbors!(sim)
            compute_gradients!(sim)
            exact &= Array(sim.grad.P) ≈ brute_force_pressure(sim)
        end
        @test exact
        @test 3 < sim.neighbors.builds < 50
        # skin = 0 rebuilds every step
        @test step!(skin_sim(0.0), 10).neighbors.builds == 10
        # extra neighbours only change the summation order
        a, b = step!(skin_sim(0.0; v0 = 0.3), 100), step!(skin_sim(0.5; v0 = 0.3), 100)
        @test b.neighbors.builds < a.neighbors.builds
        @test b.bodies.x ≈ a.bodies.x rtol = 1e-10
        @test b.bodies.coef ≈ a.bodies.coef rtol = 1e-10
    end
end
