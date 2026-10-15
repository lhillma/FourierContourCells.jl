using FourierContourCells: update_geometry!, contour_radius, node_angle, Geometry, wrap, minimum_image

@testset "geometry" begin
    @testset "basis" begin
        b = FourierBasis(; maxmode = 6, nθ = 40)
        @test nmodes(b) == 5
        @test ncoef(b) == 11
        @test size(b.C) == (40, 11)
        # columns of the synthesis matrix are orthogonal on the quadrature grid
        G = b.C' * b.C .* b.Δφ
        @test G[1, 1] ≈ 2π
        @test G[2:end, 2:end] ≈ π * I(10) atol = 1e-12
        @test_throws ArgumentError FourierBasis(; maxmode = 6, nθ = 12)
    end

    @testset "box" begin
        @test wrap(-0.5, 2.0) == (1.5, Int32(-1))
        @test wrap(4.5, 2.0) == (0.5, Int32(2))
        @test wrap(-1e-18, 2.0) == (0.0, Int32(0))
        @test minimum_image(1.6, 2.0) ≈ -0.4
    end

    @testset "contour synthesis" begin
        # one deformed, rotated body: compare matrix synthesis with direct evaluation
        maxmode, nθ = 6, 40
        coef = zeros(11, 1)
        coef[2:6, 1] = [0.1, -0.05, 0.03, 0.0, 0.02]
        coef[7:11, 1] = [0.02, 0.04, 0.0, -0.01, 0.01]
        coef[1, 1] = sqrt(1 - sum(abs2, coef[2:end, 1]) / 2)
        θ, R0 = 0.7, 1.3
        bodies = Bodies([0.0], [0.0], [θ], coef, [R0])
        basis = FourierBasis(; maxmode, nθ)
        geom = Geometry(basis, bodies)
        update_geometry!(geom, basis, bodies)
        Rdirect(φ) = R0 * (coef[1] + sum(coef[n, 1] * cos(n * (φ - θ)) + coef[5 + n, 1] * sin(n * (φ - θ)) for n in 2:6))
        φs = [node_angle(basis, k) for k in 1:nθ]
        @test geom.R[:, 1] ≈ Rdirect.(φs)
        h = 1e-6
        @test geom.dR[:, 1] ≈ (Rdirect.(φs .+ h) .- Rdirect.(φs .- h)) ./ 2h rtol = 1e-7
        @test geom.ddR[:, 1] ≈ (Rdirect.(φs .+ h) .- 2Rdirect.(φs) .+ Rdirect.(φs .- h)) ./ h^2 rtol = 1e-3
        # evaluation off the grid
        for ψ in (-2.0, 0.3, 4.0)
            @test contour_radius(geom.lab, 1, R0, 5, ψ) ≈ Rdirect(ψ)
        end
        # upper bound of R
        @test geom.rmax[1] ≥ maximum(Rdirect.(range(0, 2π, 1000)))
    end
end
