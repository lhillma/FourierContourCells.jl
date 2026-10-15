using Enzyme

# Same bending term as in test_energies.jl, but differentiated by Enzyme
struct EnzymeBending{T} <: BodyTerm
    κb::T
end
FourierContourCells.densities(::EnzymeBending, R, dR, ddR) = densities(Bending(1.0), R, dR, ddR)
FourierContourCells.combine(t::EnzymeBending, Q, b) = t.κb * Q[1]
FourierContourCells.ad_backend(::EnzymeBending) = EnzymeAD()

@testset "Enzyme extension" begin
    simF, _ = single_body_sim((Bending(0.2),))
    simE, _ = single_body_sim((EnzymeBending(0.2),))
    refresh!(simF)
    refresh!(simE)
    @test body_energy(simE) ≈ body_energy(simF)
    @test Array(simE.grad.gcoef) ≈ Array(simF.grad.gcoef)
end
