# Two contour integrals (perimeter, area) with a nonlinear energy and AD derivatives
struct PerimeterArea{T} <: BodyTerm
    λ::T
    κ::T
end
FourierContourCells.contour_integrands(::PerimeterArea, R, dR, ddR) = (sqrt(R^2 + dR^2), R^2 / 2)
FourierContourCells.energy(t::PerimeterArea, Q, b) = t.λ * (Q[1] - 6.5)^2 + t.κ * (Q[2] - π * b.R0^2)^2 * Q[1]

"Nucleated soft fluid with deformed shapes, for a given kernel layout."
function layout_sim(layout; nθ = 40)
    cells, box = hexagonal_monolayer(5, 4; packing = 0.97, rng = Xoshiro(2))
    bodies = with_nuclei(cells)
    bodies.coef[2:end, :] .= 0.04 .* randn(Xoshiro(5), size(bodies.coef, 1) - 1, nbodies(bodies))
    types = (BodyType(v0 = 0.2, Dr = 0.1), BodyType(Dr = 0.1, drag = :parent))
    energies = (Tension(0.1), OnTypes(PerimeterArea(0.05, 0.3), 1), OnTypes(PerimeterElasticity(0.2, 3.7), 2),
        Overlap([1.0 1.0; 1.0 1.0]; related = [1.0 -2.0; -2.0 1.0]), LinearRepulsion(0.5))
    return Simulation(bodies, box; types, energies, bonds = parent_bonds(bodies), nθ, layout, seed = 7)
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

    @testset "large nθ falls back to body-parallel kernels" begin
        a = refresh!(layout_sim(BodyParallel(); nθ = 300))
        b = refresh!(layout_sim(NodeParallel(); nθ = 300))
        @test b.grad.gcoef ≈ a.grad.gcoef
    end
end
