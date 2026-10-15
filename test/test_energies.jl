using FourierContourCells: ad_density_gradients, ad_combine_gradient, as_jacobian
using StaticArrays

# User-defined term used to exercise the AD path and R'' dependence:
# bending energy κ_b ∮ κ² ds with κ = (R² + 2R'² - R R'') / (R² + R'²)^{3/2}
struct Bending{T} <: BodyTerm
    κb::T
end
FourierContourCells.densities(::Bending, R, dR, ddR) = (R^2 + 2dR^2 - R * ddR)^2 / (R^2 + dR^2)^(5 / 2)
FourierContourCells.combine(t::Bending, Q, b) = t.κb * Q[1]

"Single-body simulation with one deformed body; returns (sim, coef)."
function single_body_sim(terms; θ = 0.4, R0 = 1.2, nθ = 40, type = 1, ntypes = 1)
    coef = reshape(deformed_coef(), :, 1)
    bodies = Bodies([1.0], [2.0], [θ], coef, [R0]; type)
    box = PeriodicBox(20.0, 20.0)
    sim = Simulation(bodies, box; types = ntuple(_ -> BodyType(), ntypes), energies = terms, nθ)
    return sim, coef
end

"Central finite-difference gradient of the total body energy w.r.t. shape modes."
function fd_shape_gradient(sim, coef; h = 1e-6)
    g = zeros(size(coef, 1) - 1)
    for r in 2:size(coef, 1)
        E = map((h, -h)) do δ
            c = copy(coef)
            c[r] += δ
            copyto!(sim.bodies.coef, c)
            refresh!(sim)
            body_energy(sim)
        end
        g[r - 1] = (E[1] - E[2]) / 2h
    end
    copyto!(sim.bodies.coef, coef)
    refresh!(sim)
    return g
end

@testset "energies" begin
    @testset "manual vs AD derivatives (Tension)" begin
        t = Tension(0.3)
        for (R, dR, ddR) in ((1.0, 0.0, 0.0), (0.9, 0.2, -0.5), (1.3, -0.4, 0.1))
            @test as_jacobian(density_gradients(t, R, dR, ddR)) ≈ ad_density_gradients(ForwardDiffAD(), t, R, dR, ddR)
        end
        Q = SVector(6.5)
        @test combine_gradient(t, Q, BodyInfo(1.0, Int32(1))) ≈ ad_combine_gradient(ForwardDiffAD(), t, Q, BodyInfo(1.0, Int32(1)))
    end

    @testset "circle energies" begin
        sim, _ = single_body_sim((Tension(0.1),); R0 = 1.0, nθ = 64)
        copyto!(sim.bodies.coef, [1.0; zeros(10)])
        refresh!(sim)
        @test body_energy(sim) ≈ 0.1 * 2π
        @test shape_index(sim)[1] ≈ 2sqrt(π)
    end

    @testset "finite-difference gradient: $name" for (name, term) in (
            ("Tension", Tension(0.37)),
            ("PerimeterElasticity (AD)", PerimeterElasticity(0.5, 3.9)),
            ("Bending (user term, AD, uses R'')", Bending(0.2)),
        )
        sim, coef = single_body_sim((term,))
        refresh!(sim)
        g = Array(sim.grad.gcoef)[2:end, 1]
        @test g ≈ fd_shape_gradient(sim, coef) rtol = 1e-6
        # single-body terms exert no force or torque
        @test sim.grad.gx[1] == 0 && sim.grad.gy[1] == 0 && sim.grad.gθ[1] == 0
    end

    @testset "OnTypes restricts terms" begin
        for (type, active) in ((1, false), (2, true))
            sim, _ = single_body_sim((OnTypes(Tension(1.0), 2),); type, ntypes = 2)
            refresh!(sim)
            @test (body_energy(sim) > 0) == active
            @test any(!iszero, sim.grad.gcoef) == active
        end
    end
end
