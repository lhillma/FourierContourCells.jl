"""
    Tension(η)

Line tension η·l with the perimeter l = ∮ √(R² + R'²) dφ (Saito & Ishihara,
2024). Derivatives are implemented by hand.
"""
struct Tension{T} <: BodyTerm
    η::T
end

@inline contour_integrands(::Tension, R, dR, ddR) = sqrt(R^2 + dR^2)
@inline energy(t::Tension, Q, b) = t.η * Q[1]
@inline function contour_integrand_gradients(::Tension, R, dR, ddR)
    s = sqrt(R^2 + dR^2)
    return (R / s, dR / s, zero(s))
end
@inline energy_gradient(t::Tension, Q, b) = SVector(t.η)

"""
    PerimeterElasticity(λ, p0)

CPM-like perimeter elasticity λ (l - l₀)² with the target perimeter
l₀ = p0·√(πR0²) set by the target shape index `p0` (p0 = 2√π ≈ 3.545 for a
circle). Derivatives come from automatic differentiation; this term doubles as
a template for user-defined terms.
"""
struct PerimeterElasticity{T} <: BodyTerm
    λ::T
    p0::T
end

@inline contour_integrands(::PerimeterElasticity, R, dR, ddR) = sqrt(R^2 + dR^2)
@inline energy(t::PerimeterElasticity, Q, b) = t.λ * (Q[1] - t.p0 * sqrt(oftype(t.p0, π)) * b.R0)^2
