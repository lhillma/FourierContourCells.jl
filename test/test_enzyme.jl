using Enzyme

# Same bending term as in test_energies.jl, but differentiated by Enzyme
struct EnzymeBending{T} <: BodyTerm
    κb::T
end
FourierContourCells.contour_integrands(::EnzymeBending, R, dR, ddR) = contour_integrands(Bending(1.0), R, dR, ddR)
FourierContourCells.energy(t::EnzymeBending, Q, b) = t.κb * Q[1]
FourierContourCells.ad_backend(::EnzymeBending) = EnzymeAD()

# Quartic bond of test_parents.jl, differentiated by Enzyme
struct EnzymeQuarticBond{T} <: BondTerm
    c::T
end
FourierContourCells.bond_energy(b::EnzymeQuarticBond, Δ, ti, tj) = bond_energy(QuarticBond(b.c), Δ, ti, tj)
FourierContourCells.ad_backend(::EnzymeQuarticBond) = EnzymeAD()

@testset "Enzyme extension" begin
    simF, _ = single_body_sim((Bending(0.2),))
    simE, _ = single_body_sim((EnzymeBending(0.2),))
    refresh!(simF)
    refresh!(simE)
    @test body_energy(simE) ≈ body_energy(simF)
    @test Array(simE.grad.gcoef) ≈ Array(simF.grad.gcoef)
    Δ = SVector(0.3, -0.7)
    @test bond_gradient(EnzymeQuarticBond(0.8), Δ, 1, 2) ≈ bond_gradient(QuarticBond(0.8), Δ, 1, 2)
end
