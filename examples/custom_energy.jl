# Template for user-defined energy terms.
#
#   julia -t auto --project=examples examples/custom_energy.jl
#
# Single-body terms are written as E = F(Q₁, …, Qₘ) with contour integrals
# Qⱼ = ∮ qⱼ(R, R', R'') dφ: define `contour_integrands` (the qⱼ) and `energy` (F).
# Derivatives come from automatic differentiation (ForwardDiff by default;
# `ad_backend(::MyTerm) = EnzymeAD()` after `using Enzyme`), or can be written
# by hand by adding `contour_integrand_gradients` and `energy_gradient` methods.
#
# Pair terms define the energy density g(d) felt by a contour point at
# penetration depth d into a neighbour (d > 0 inside). No derivative is needed.

using FourierContourCells
import FourierContourCells: contour_integrands, energy, pressure, interaction_range
using GLMakie

# Bending rigidity κ ∮ curvature² ds, curvature from R, R', R''.
struct Bending{T} <: BodyTerm
    κ::T
end
contour_integrands(::Bending, R, dR, ddR) = (R^2 + 2dR^2 - R * ddR)^2 / (R^2 + dR^2)^(5 / 2)
energy(t::Bending, Q, body) = t.κ * Q[1]

# Short-range adhesion: area of a cell lying within a band of width w outside a
# neighbour lowers the energy by γ per unit area.
struct BandAdhesion{T} <: ContactTerm
    γ::T
    w::T
end
pressure(a::BandAdhesion, d, ti, tj, related) = ifelse(-a.w <= d < 0, -oftype(d, a.γ), zero(d))
interaction_range(a::BandAdhesion) = a.w

bodies, box = hexagonal_monolayer(16, 16; packing = 0.7)
sim = Simulation(bodies, box;
    types = (BodyType(v0 = 0.03, Dr = 0.01, μab = 0.1),),
    energies = (Tension(0.01), Bending(0.005), Overlap(1.0), BandAdhesion(0.05, 0.1)))

FourierContourCells.live_view(sim; steps_per_frame = 50)
