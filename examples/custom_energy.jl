# Tutorial: user-defined energy terms
#
#   julia -t auto --project=examples examples/custom_energy.jl
#
# Energy terms never see Fourier coefficients. They are written in terms of the
# cell contour in polar form around the cell centre, R(φ), and its derivatives
# R' = dR/dφ and R'' = d²R/dφ². The library converts gradients with respect to
# R, R', R'' into forces, torques and shape-mode forces by the chain rule.
#
# There are two kinds of terms:
#
#   BodyTerm     energy of a single cell from its shape    (sections 1–3)
#   ContactTerm  interaction between neighbouring cells    (section 4)

using FourierContourCells
using GLMakie
import FourierContourCells: contour_integrands, energy, contour_integrand_gradients,
    pressure, interaction_range

# ---------------------------------------------------------------------------
# 1. The simplest BodyTerm: an energy density integrated around the contour
#
#        E = ∮ e(R, R', R'') dφ
#
#    Define `contour_integrands` returning e. That is all: the library sums
#    e over the contour nodes (times Δφ) and differentiates it with ForwardDiff.
#
#    Example: bending rigidity κ ∮ c² ds, with the curvature
#    c = (R² + 2R'² - R R'') / (R² + R'²)^{3/2} and the arc length element
#    ds = √(R² + R'²) dφ, so e = κ c² √(R² + R'²).

struct Bending{T} <: BodyTerm
    κ::T
end

contour_integrands(t::Bending, R, dR, ddR) = t.κ * (R^2 + 2dR^2 - R * ddR)^2 / (R^2 + dR^2)^(5 / 2)

# ---------------------------------------------------------------------------
# 2. Nonlinear energies of contour integrals
#
#        E = F(Q₁, …, Qₘ),   Qⱼ = ∮ qⱼ(R, R', R'') dφ
#
#    `contour_integrands` returns the integrands (q₁, …, qₘ) as a tuple; the
#    library integrates them and passes the integrals Q (an SVector) to
#    `energy`, which returns F(Q). `body` carries per-cell data: body.R0 (radius
#    scale, area πR0²) and body.type.
#
#    Example: CPM-like perimeter elasticity plus a roundness penalty. Q₁ is the
#    perimeter, Q₂ = ∮ R'² dφ measures the deviation from a circle.

struct PerimeterAndRoundness{T} <: BodyTerm
    λ::T     # perimeter elasticity
    p0::T    # target shape index (perimeter / √area); a circle has 2√π ≈ 3.545
    γ::T     # roundness stiffness
end

contour_integrands(::PerimeterAndRoundness, R, dR, ddR) = (sqrt(R^2 + dR^2), dR^2)

function energy(t::PerimeterAndRoundness, Q, body)
    perimeter, roundness = Q[1], Q[2]
    target = t.p0 * sqrt(oftype(t.p0, π)) * body.R0
    return t.λ * (perimeter - target)^2 + t.γ * roundness
end

# ---------------------------------------------------------------------------
# 3. Optional: derivatives by hand
#
#    AD is exact and usually fast enough. For hot terms you can supply the
#    derivatives yourself; they take precedence over AD:
#
#      contour_integrand_gradients(t, R, dR, ddR)
#          ∂qⱼ/∂(R, R', R''): a 3-tuple for one integrand, a tuple of 3-tuples
#          for several (one per integrand), or an m × 3 SMatrix
#      energy_gradient(t, Q, body)
#          ∂F/∂Qⱼ: a Number, Tuple or SVector
#
#    Here for the bending term (one integrand; energy is the default sum(Q),
#    whose gradient is known, so only the integrand gradient is needed):

function contour_integrand_gradients(t::Bending, R, dR, ddR)
    s2 = R^2 + dR^2
    num = R^2 + 2dR^2 - R * ddR
    ∂R = t.κ * (2num * (2R - ddR) / s2^(5 / 2) - 5num^2 * R / s2^(7 / 2))
    ∂dR = t.κ * (8num * dR / s2^(5 / 2) - 5num^2 * dR / s2^(7 / 2))
    ∂ddR = t.κ * (-2num * R / s2^(5 / 2))
    return (∂R, ∂dR, ∂ddR)
end

# To use Enzyme instead of ForwardDiff for a term (after `using Enzyme`):
#     FourierContourCells.ad_backend(::MyTerm) = EnzymeAD()

# ---------------------------------------------------------------------------
# 4. ContactTerm: interactions through the contour
#
#    For every contour point u of cell i and every neighbour j, the library
#    computes the penetration depth d = Rⱼ(ψ) - |u - rⱼ| of u into j (d > 0:
#    inside j; d < 0: outside, at radial distance |d| from j's contour).
#    `pressure(term, d, tᵢ, tⱼ, related)` returns the energy per unit area g(d)
#    that the area of cell i feels there; tᵢ, tⱼ are the body types and
#    `related` tells whether both bodies share a parent (cell and its nucleus).
#    The energy is Eᵢ = Σⱼ ∫_{cell i} g(dⱼ) dA and needs no derivative.
#    If g ≠ 0 outside contact (d < 0), declare how far with `interaction_range`.
#
#    Example: short-range adhesion: area of a cell within a band of width w
#    outside a neighbour lowers the energy by γ per unit area.

struct BandAdhesion{T} <: ContactTerm
    γ::T
    w::T
end

pressure(a::BandAdhesion, d, ti, tj, related) = ifelse(-a.w <= d < 0, -oftype(d, a.γ), zero(d))
interaction_range(a::BandAdhesion) = a.w

# ---------------------------------------------------------------------------
# Run: terms are simply listed; body and contact terms are sorted out
# automatically. OnTypes(term, types...) restricts a term to some body types.

bodies, box = hexagonal_monolayer(16, 16; packing = 0.7)
sim = Simulation(bodies, box;
    types = (BodyType(v0 = 0.03, Dr = 0.01, μab = 0.1),),
    energies = (Tension(0.01), Bending(0.005), PerimeterAndRoundness(0.01, 3.7, 0.001),
        Overlap(1.0), BandAdhesion(0.05, 0.1)))

FourierContourCells.live_view(sim; steps_per_frame = 50)
