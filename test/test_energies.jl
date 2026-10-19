using FourierContourCells: ad_contour_integrand_gradients, ad_energy_gradient, as_jacobian
using StaticArrays

# User-defined term used to exercise the AD path and R'' dependence:
# bending energy κ_b ∮ κ² ds with κ = (R² + 2R'² - R R'') / (R² + R'²)^{3/2}
struct Bending{T} <: BodyTerm
    κb::T
end
FourierContourCells.contour_integrands(::Bending, R, dR, ddR) = (R^2 + 2dR^2 - R * ddR)^2 / (R^2 + dR^2)^(5 / 2)
FourierContourCells.energy(t::Bending, Q, b) = t.κb * Q[1]

# Integrand-only term: energy defaults to sum(Q), i.e. η ∮ √(R² + R'²) dφ
struct DensityOnlyTension{T} <: BodyTerm
    η::T
end
FourierContourCells.contour_integrands(t::DensityOnlyTension, R, dR, ddR) = t.η * sqrt(R^2 + dR^2)

# Term whose constructor does not take its fields, with the documented
# convert_precision override (stores η and the derived 2η)
struct DoubledTension{T} <: BodyTerm
    η::T
    twoη::T
    DoubledTension(η::T) where {T} = new{T}(η, 2η)
end
FourierContourCells.contour_integrands(t::DoubledTension, R, dR, ddR) = t.η * sqrt(R^2 + dR^2)
FourierContourCells.convert_precision(::Type{T}, t::DoubledTension) where {T} = DoubledTension(T(t.η))

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
            @test as_jacobian(contour_integrand_gradients(t, R, dR, ddR)) ≈ ad_contour_integrand_gradients(ForwardDiffAD(), t, R, dR, ddR)
        end
        Q = SVector(6.5)
        @test energy_gradient(t, Q, BodyInfo(1.0, Int32(1))) ≈ ad_energy_gradient(ForwardDiffAD(), t, Q, BodyInfo(1.0, Int32(1)))
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
            ("integrand-only term (default energy)", DensityOnlyTension(0.37)),
        )
        sim, coef = single_body_sim((term,))
        refresh!(sim)
        g = Array(sim.grad.gcoef)[2:end, 1]
        @test g ≈ fd_shape_gradient(sim, coef) rtol = 1e-6
        # single-body terms exert no force or torque
        @test sim.grad.gx[1] == 0 && sim.grad.gy[1] == 0 && sim.grad.gθ[1] == 0
    end

    @testset "default energy is the sum of contour integrals" begin
        simA, _ = single_body_sim((Tension(0.37),))
        simB, _ = single_body_sim((DensityOnlyTension(0.37),))
        refresh!(simA)
        refresh!(simB)
        @test body_energy(simB) ≈ body_energy(simA)
        @test Array(simB.grad.gcoef) ≈ Array(simA.grad.gcoef)
    end

    @testset "OnTypes restricts terms" begin
        for (type, active) in ((1, false), (2, true))
            sim, _ = single_body_sim((OnTypes(Tension(1.0), 2),); type, ntypes = 2)
            refresh!(sim)
            @test (body_energy(sim) > 0) == active
            @test any(!iszero, sim.grad.gcoef) == active
        end
    end

    @testset "term parameters follow the simulation precision" begin
        cp = FourierContourCells.convert_precision
        @test cp(Float32, Tension(0.01)) === Tension(0.01f0)
        @test cp(Float32, HarmonicBond(0.2; r0 = 0.1)) === HarmonicBond(0.2f0, 0.1f0)
        o = cp(Float32, Overlap([1.0 2.0; 2.0 3.0]; related = [1.0 -0.5; -0.5 1.0]))
        @test eltype(o.J) == eltype(o.Jrelated) == Float32
        @test o.Jrelated[1, 2] == -0.5f0
        w = cp(Float32, OnTypes(PerimeterElasticity(0.2, 3.7), 2))
        @test w.term === PerimeterElasticity(0.2f0, 3.7f0)
        @test w.types === (Int32(2),)
        @test cp(Float64, Tension(0.01f0)) === Tension(Float64(0.01f0))
        @test cp(Float32, Bending(0.5)).κb isa Float32
        # a constructor that does not take the fields: unchanged, with a warning
        struct FixedTerm <: BodyTerm
            a::Float64
            FixedTerm() = new(1.0)
        end
        @test (@test_logs (:warn,) cp(Float32, FixedTerm())) === FixedTerm()
        @test (@test_logs cp(Float64, FixedTerm())) === FixedTerm()  # nothing to convert: no warning
        # a user method applies alone and inside OnTypes (no ambiguity, no warning)
        @test (@test_logs cp(Float32, DoubledTension(0.1))) === DoubledTension(0.1f0)
        @test (@test_logs cp(Float32, OnTypes(DoubledTension(0.1), 1))).term === DoubledTension(0.1f0)
        sim = Simulation(hexagonal_monolayer(2, 2)...; types = (BodyType(),),
            energies = (Tension(0.01), Overlap(1.0)), T = Float32)
        @test sim.body_terms[1] === Tension(0.01f0)
        @test eltype(sim.contact_terms[1].J) == Float32
    end
end
