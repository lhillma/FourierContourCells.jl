# Energy-term interface
#
# Two kinds of terms exist. Both are differentiated with respect to the contour
# samples R(φₖ), R'(φₖ), R''(φₖ) at the quadrature nodes (and the body centre);
# the library maps these contour gradients to the dynamical variables
# (r, θ, aₙ, bₙ) via the chain rule, see `gradients.jl`. Users therefore never
# deal with Fourier coefficients.
#
# 1. `BodyTerm`: energy of a single body built from contour integrals,
#        E = F(Q₁, …, Qₘ),   Qⱼ = ∮ qⱼ(R, R', R'') dφ.
#    Users define the integrands qⱼ (`contour_integrands`) and the function F
#    (`energy`). Derivatives come from automatic differentiation unless
#    `contour_integrand_gradients` / `energy_gradient` are defined by hand.
#
# 2. `ContactTerm`: pair interaction acting on a body through its own contour.
#    Users define the energy density `pressure(term, d, tᵢ, tⱼ, related)` felt by
#    a contour point of body i at penetration depth d into body j; the energy
#    of body i is Eᵢ = Σⱼ ∫_{Ωᵢ} g(dⱼ(x)) dx and its shape derivative is
#    g evaluated on the contour. No derivative of g is needed.
#
# 3. `BondTerm`: pair energy E(rᵢ - rⱼ) between the centres of explicitly
#    listed body pairs (`Simulation(...; bonds)`), e.g. a spring between a
#    cell and its nucleus. Users define `bond_energy`; its gradient comes from
#    AD unless `bond_gradient` is defined by hand.

"Supertype of all energy terms."
abstract type EnergyTerm end

"""
    BodyTerm <: EnergyTerm

Energy of a single body that depends on its shape only. It is written in two
stages, which keeps user code small while allowing nonlinear energies:

1. **Contour integrals.** [`contour_integrands`](@ref) returns the integrands
   q₁, …, qₘ as functions of the local contour radius R(φ) and its derivatives
   R' = dR/dφ, R'' = d²R/dφ². The library integrates them around the contour:
   Qⱼ = ∮ qⱼ(R, R', R'') dφ. Useful integrands:

   | quantity   | integrand q(R, R', R'')                       |
   |:-----------|:----------------------------------------------|
   | perimeter  | √(R² + R'²)                                   |
   | area       | R²/2                                          |
   | ∮ κ² ds    | (R² + 2R'² - R R'')² / (R² + R'²)^(5/2)        |

2. **Energy.** [`energy`](@ref) maps the integrals Q = (Q₁, …, Qₘ) (an
   `SVector`) and per-body data [`BodyInfo`](@ref) (`R0`, `type`) to the
   energy E = F(Q). It defaults to `sum(Q)`, so a term whose integrand already
   is an energy density (E = ∮ e dφ) only needs `contour_integrands`.

Example, a CPM-like perimeter elasticity λ(l - l₀)²:

    struct MyPerimeter{T} <: BodyTerm
        λ::T
        l0::T
    end
    FourierContourCells.contour_integrands(::MyPerimeter, R, dR, ddR) = sqrt(R^2 + dR^2)  # Q[1] = l
    FourierContourCells.energy(t::MyPerimeter, Q, body) = t.λ * (Q[1] - t.l0)^2

Derivatives are obtained by automatic differentiation (backend chosen by
[`ad_backend`](@ref)). For speed they can be provided by hand:

    contour_integrand_gradients(term, R, dR, ddR) -> ∂qⱼ/∂(R, R', R''): an m × 3
        SMatrix, a 3-tuple (m = 1) or a tuple of 3-tuples
    energy_gradient(term, Q, body) -> ∂F/∂Qⱼ as SVector, Tuple or Number

[`applies_to`](@ref)`(term, type)` can restrict a term to body types (or wrap
it in [`OnTypes`](@ref)). All methods must be allocation-free and type-stable,
as they run inside (GPU) kernels.
"""
abstract type BodyTerm <: EnergyTerm end

"""
    ContactTerm <: EnergyTerm

Pair interaction expressed as an energy density on the contour. Required:

    pressure(term, d, tᵢ, tⱼ, related) -> g(d)

where `d = Rⱼ(ψ) - |u - rⱼ|` is the radial penetration depth of the contour
point `u` of body i into body j (d > 0: inside), `tᵢ`, `tⱼ` are the body types
and `related` tells whether both bodies have the same parent. Optional:

    interaction_range(term) -> largest |d| with d < 0 at which g ≠ 0 (default 0)
"""
abstract type ContactTerm <: EnergyTerm end

"""
    BondTerm <: EnergyTerm

Pair energy between the centres of bonded bodies (bond list passed to
[`Simulation`](@ref) as `bonds`; see [`parent_bonds`](@ref)). Required:

    bond_energy(term, Δ, tᵢ, tⱼ) -> E

with `Δ` = rᵢ - rⱼ (an `SVector{2}`, minimum image) and the body types `tᵢ`,
`tⱼ`. The energy must be symmetric, E(Δ, tᵢ, tⱼ) = E(-Δ, tⱼ, tᵢ), as each
body of a bond evaluates it from its own side. Optional:

    bond_gradient(term, Δ, tᵢ, tⱼ) -> ∂E/∂Δ as SVector{2}

(default: automatic differentiation). Bond energies are not included in
[`body_energy`](@ref). Wrapped in [`OnTypes`](@ref), a bond term acts on bonds
whose bodies both have one of the listed types.
"""
abstract type BondTerm <: EnergyTerm end

"""
    bond_energy(term::BondTerm, Δ, tᵢ, tⱼ) -> E

Energy of a bond with centre separation `Δ = rᵢ - rⱼ` (see [`BondTerm`](@ref)).
"""
function bond_energy end

"""
    BodyInfo(R0, type)

Per-body data passed to [`energy`](@ref): the radius scale `R0` (e.g. for a
target area πR0²) and the body `type`.
"""
struct BodyInfo{T}
    R0::T
    type::Int32
end

"""
    contour_integrands(term::BodyTerm, R, dR, ddR) -> q or (q₁, …, qₘ)

Integrands of the contour integrals Qⱼ = ∮ qⱼ dφ of a [`BodyTerm`](@ref), as
functions of the contour radius `R`, `dR` = dR/dφ and `ddR` = d²R/dφ².
"""
function contour_integrands end

"""
    energy(term::BodyTerm, Q::SVector, body::BodyInfo) -> E

Energy of one body from its contour integrals `Q` (see [`BodyTerm`](@ref)).
Defaults to `sum(Q)`.
"""
energy(::BodyTerm, Q, body) = sum(Q)

"""
    pressure(term::ContactTerm, d, tᵢ, tⱼ, related) -> g

Energy density of a [`ContactTerm`](@ref) at penetration depth `d`.
"""
function pressure end

"Whether `term` acts on bodies of type `type` (default: all types)."
applies_to(::EnergyTerm, type) = true

"Range beyond contact (d < 0) over which a [`ContactTerm`](@ref) acts."
interaction_range(::ContactTerm) = 0

## Automatic differentiation backends

"Supertype of AD backends used for [`BodyTerm`](@ref) derivatives."
abstract type ADBackend end

"Forward-mode AD with ForwardDiff.jl dual numbers (default; GPU-compatible)."
struct ForwardDiffAD <: ADBackend end

"""
Forward-mode AD with Enzyme.jl. Requires `using Enzyme` to load the
`FourierContourCellsEnzymeExt` extension.
"""
struct EnzymeAD <: ADBackend end

"""
    ad_backend(term) -> ADBackend

AD backend used for `term` when no manual derivative is defined. Override per
term type, e.g. `ad_backend(::MyTerm) = EnzymeAD()`.
"""
ad_backend(::EnergyTerm) = ForwardDiffAD()

contour_integrand_gradients(term::BodyTerm, R, dR, ddR) = ad_contour_integrand_gradients(ad_backend(term), term, R, dR, ddR)
energy_gradient(term::BodyTerm, Q, b) = ad_energy_gradient(ad_backend(term), term, Q, b)

"""
    bond_gradient(term::BondTerm, Δ, tᵢ, tⱼ) -> SVector{2}

Gradient ∂E/∂Δ of [`bond_energy`](@ref); automatic differentiation by default.
"""
bond_gradient(term::BondTerm, Δ, ti, tj) = ad_bond_gradient(ad_backend(term), term, Δ, ti, tj)

function ad_contour_integrand_gradients(::ForwardDiffAD, term, R, dR, ddR)
    f(v) = as_svector(contour_integrands(term, v[1], v[2], v[3]))
    return ForwardDiff.jacobian(f, SVector(R, dR, ddR))
end

ad_energy_gradient(::ForwardDiffAD, term, Q, b) = ForwardDiff.gradient(q -> energy(term, q, b), Q)

ad_bond_gradient(::ForwardDiffAD, term, Δ, ti, tj) = ForwardDiff.gradient(d -> bond_energy(term, d, ti, tj), Δ)

ad_contour_integrand_gradients(::EnzymeAD, args...) = enzyme_missing()
ad_energy_gradient(::EnzymeAD, args...) = enzyme_missing()
ad_bond_gradient(::EnzymeAD, args...) = enzyme_missing()
enzyme_missing() = error("EnzymeAD requires `using Enzyme` to load the Enzyme extension")

## Normalisation of user return values

@inline as_svector(x::Number) = SVector(x)
@inline as_svector(x::Tuple) = SVector(x)
@inline as_svector(x::SVector) = x

@inline as_jacobian(J::SMatrix) = J
@inline as_jacobian(g::NTuple{3, Number}) = SMatrix{1, 3}(promote(g...))
@inline as_jacobian(g::Tuple{Vararg{NTuple{3, Number}}}) = vcat(map(as_jacobian, g)...)

## Restriction to body types

"""
    OnTypes(term, types...)

Restrict `term` to bodies whose type is in `types`. For a [`ContactTerm`](@ref)
the restriction applies to the body whose contour is evaluated (body i); a
[`BondTerm`](@ref) acts on a bond if both bodies have a listed type.
"""
struct OnTypes{Tm <: EnergyTerm, N} <: EnergyTerm
    term::Tm
    types::NTuple{N, Int32}
end
OnTypes(term::EnergyTerm, types::Integer...) = OnTypes(term, Int32.(types))

applies_to(t::OnTypes, type) = type in t.types && applies_to(t.term, type)

# OnTypes is a BodyTerm/ContactTerm in behaviour depending on the wrapped term
is_body_term(t) = t isa BodyTerm || (t isa OnTypes && is_body_term(t.term))
is_contact_term(t) = t isa ContactTerm || (t isa OnTypes && is_contact_term(t.term))
is_bond_term(t) = t isa BondTerm || (t isa OnTypes && is_bond_term(t.term))

contour_integrands(t::OnTypes, R, dR, ddR) = contour_integrands(t.term, R, dR, ddR)
energy(t::OnTypes, Q, b) = energy(t.term, Q, b)
contour_integrand_gradients(t::OnTypes, R, dR, ddR) = contour_integrand_gradients(t.term, R, dR, ddR)
energy_gradient(t::OnTypes, Q, b) = energy_gradient(t.term, Q, b)
pressure(t::OnTypes, d, ti, tj, rel) = pressure(t.term, d, ti, tj, rel)
interaction_range(t::OnTypes) = interaction_range(t.term)
bond_energy(t::OnTypes, Δ, ti, tj) = bond_energy(t.term, Δ, ti, tj)
bond_gradient(t::OnTypes, Δ, ti, tj) = bond_gradient(t.term, Δ, ti, tj)
