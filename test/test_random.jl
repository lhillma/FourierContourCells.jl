using FourierContourCells: philox4x32, philox_key, philox_noise!, philox_normals

@testset "Philox random numbers" begin
    @testset "known-answer vectors (Random123)" begin
        @test philox4x32((0x0, 0x0, 0x0, 0x0) .% UInt32, (0x0, 0x0) .% UInt32) ==
            (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
        @test philox4x32((0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), (0xffffffff, 0xffffffff)) ==
            (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)
        @test philox4x32((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0)) ==
            (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)
    end

    @testset "standard normal noise ($T)" for T in (Float32, Float64)
        key = philox_key(42)
        ξ = philox_noise!(zeros(T, 3, 100_000), key, 7)
        @test eltype(ξ) == T
        for r in 1:3
            @test abs(mean(ξ[r, :])) < 0.015
            @test var(ξ[r, :]) ≈ 1 atol = 0.02
            @test mean(ξ[r, :] .^ 4) ≈ 3 atol = 0.1
        end
        @test abs(cor(ξ[1, :], ξ[2, :])) < 0.015
        @test abs(cor(ξ[1, :], ξ[3, :])) < 0.015
        # next draw is independent, a body's numbers do not depend on N
        ξ8 = philox_noise!(zeros(T, 3, 100_000), key, 8)
        @test abs(cor(vec(ξ), vec(ξ8))) < 0.01
        @test philox_noise!(zeros(T, 3, 10), key, 7) == ξ[:, 1:10]
        @test philox_noise!(zeros(T, 3, 10), philox_key(43), 7) != ξ[:, 1:10]
        @test all(isfinite, ξ)
    end

    @testset "draw index uses all 64 bits" begin
        key = philox_key(1)
        @test philox_normals(Float64, key, 1, 1) != philox_normals(Float64, key, 1, 1 + (1 << 32))
    end

    @testset "same seed, same trajectory" begin
        sim(seed) = Simulation(hexagonal_monolayer(4, 4)...; types = (BodyType(v0 = 0.1, Dr = 0.5, Dt = 0.01),),
            energies = (Tension(0.1), Overlap(1.0)), seed)
        a, b, c = step!(sim(3), 20), step!(sim(3), 20), step!(sim(4), 20)
        @test a.bodies.θ == b.bodies.θ
        @test a.bodies.θ != c.bodies.θ
    end
end
