# Two contour integrals (perimeter, area) with a nonlinear energy and AD derivatives
struct PerimeterArea{T} <: BodyTerm
    λ::T
    κ::T
end
FourierContourCells.contour_integrands(::PerimeterArea, R, dR, ddR) = (sqrt(R^2 + dR^2), R^2 / 2)
FourierContourCells.energy(t::PerimeterArea, Q, b) = t.λ * (Q[1] - 6.5)^2 + t.κ * (Q[2] - π * b.R0^2)^2 * Q[1]

# Integrand defined only for R > r: e.g. on unit cells but not on nuclei of radius 0.5
struct OutsideRadius{T} <: BodyTerm
    κ::T
    r::T
end
FourierContourCells.contour_integrands(t::OutsideRadius, R, dR, ddR) = t.κ * sqrt(R^2 - t.r^2)

const LAYOUT_ENERGIES = (Tension(0.1), OnTypes(PerimeterArea(0.05, 0.3), 1),
    OnTypes(PerimeterElasticity(0.2, 3.7), 2), Overlap([1.0 1.0; 1.0 1.0]; related = [1.0 -2.0; -2.0 1.0]),
    LinearRepulsion(0.5))

"Nucleated soft fluid with deformed shapes, for a given kernel layout."
function layout_sim(layout; nθ = 40, backend = CPU(), T = Float64, v0 = 0.2, skin = 0.3, seed = 7,
        energies = LAYOUT_ENERGIES, kw...)
    cells, box = hexagonal_monolayer(5, 4; packing = 0.97, rng = Xoshiro(2))
    bodies = with_nuclei(cells)
    bodies.coef[2:end, :] .= 0.04 .* randn(Xoshiro(5), size(bodies.coef, 1) - 1, nbodies(bodies))
    types = (BodyType(; v0, Dr = 0.1), BodyType(Dr = 0.1, drag = :parent))
    return Simulation(bodies, box; types, energies, bonds = parent_bonds(bodies), nθ, layout, seed,
        backend, T, skin, kw...)
end

@testset "kernel layouts" begin
    @test FourierContourCells.default_layout(CPU()) == BodyParallel()
    @test layout_sim(BodyParallel()).layout == BodyParallel()

    @testset "same gradients (nθ = $nθ)" for nθ in (40, 37)
        a = refresh!(layout_sim(BodyParallel(); nθ))
        b = refresh!(layout_sim(NodeParallel(); nθ))
        # 40 bodies do not fill whole workgroups of 256 ÷ nθ bodies
        @test nbodies(a) % (256 ÷ nθ) != 0
        @test count(>(0), a.grad.P) > 100
        for f in (:GR, :GD, :GDD, :P, :gx, :gy, :gθ, :gcoef, :Ebody)
            @test getfield(b.grad, f) ≈ getfield(a.grad, f) rtol = 1e-12 atol = 1e-12
        end
    end

    @testset "same trajectory" begin
        a = layout_sim(BodyParallel())
        b = layout_sim(NodeParallel())
        step!(a, 50)
        step!(b, 50)
        @test b.bodies.x ≈ a.bodies.x
        @test b.bodies.coef ≈ a.bodies.coef
    end

    @testset "fused contact kernel matches pressure + gradient kernels" begin
        sim = refresh!(layout_sim(BodyParallel()))
        g = FourierContourCells.reset!(sim.grad)
        FourierContourCells.contact_gradients!(g, sim.contact_terms, sim.geom, sim.bodies,
            sim.neighbors, sim.box, sim.basis)
        fused = map(copy, (g.P, g.GR, g.gx, g.gy, g.gθ))
        FourierContourCells.reset!(g)
        FourierContourCells.contact_pressure!(g.P, sim.contact_terms, sim.geom, sim.bodies,
            sim.neighbors, sim.box, sim.basis)
        FourierContourCells.contact_gradient!(g, sim.geom, sim.basis)
        for (a, b) in zip(fused, (g.P, g.GR, g.gx, g.gy, g.gθ))
            @test a ≈ b rtol = 1e-12 atol = 1e-12
        end
    end

    @testset "excluded terms are not evaluated" begin
        # the integrand would throw a DomainError on the nuclei (R ≈ 0.5)
        energies = (Tension(0.1), OnTypes(OutsideRadius(0.05, 0.6), 1), Overlap([1.0 1.0; 1.0 1.0]))
        a = refresh!(layout_sim(BodyParallel(); energies))
        b = refresh!(layout_sim(NodeParallel(); energies))
        @test b.grad.gcoef ≈ a.grad.gcoef rtol = 1e-12 atol = 1e-12
        @test b.grad.Ebody ≈ a.grad.Ebody rtol = 1e-12
    end

    @testset "integrand counts are inferred, not evaluated" begin
        n = FourierContourCells.n_integrands
        @test n(Tension(0.1), Float64) == 1
        @test n(PerimeterArea(0.1, 0.2), Float32) == 2
        @test n(OnTypes(PerimeterArea(0.1, 0.2), 1), Float64) == 2
        @test n(OutsideRadius(1.0, 2.0), Float64) == 1  # sqrt(1 - 4) would throw
    end

    @testset "large nθ falls back to body-parallel kernels" begin
        a = refresh!(layout_sim(BodyParallel(); nθ = 300))
        b = refresh!(layout_sim(NodeParallel(); nθ = 300))
        @test b.grad.gcoef ≈ a.grad.gcoef
    end
end
