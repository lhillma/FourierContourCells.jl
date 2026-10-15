# Energy-term interface
#
# Two kinds of terms exist, both differentiated with respect to the contour
# samples R(φₖ), R'(φₖ), R''(φₖ) (and the body centre). The library maps these
# contour gradients to the dynamical variables (r, θ, aₙ, bₙ) via the chain
# rule, see `gradients.jl`.
#
# 1. `BodyTerm`: single-body energy E = F(Q₁, …, Qₘ) with contour integrals
#        Qⱼ = ∮ qⱼ(R, R', R'') dφ.
#    Users define `densities` (the qⱼ) and `combine` (F). Derivatives come from
#    automatic differentiation unless `density_gradients` / `combine_gradient`
#    are defined by hand.
#
# 2. `ContactTerm`: pair interaction acting on a body through its own contour.
#    Users define the energy density `pressure(term, d, tᵢ, tⱼ, related)` felt by
#    a contour point of body i at penetration depth d into body j; the energy
#    of body i is Eᵢ = Σⱼ ∫_{Ωᵢ} g(dⱼ(x)) dx and its shape derivative is
#    g evaluated on the contour. No derivative of g is needed.

"Supertype of all energy terms."
abstract type EnergyTerm end

"""
    BodyTerm <: EnergyTerm

Single-body energy E = F(Q₁, …, Qₘ), Qⱼ = ∮ qⱼ(R, R', R'') dφ. Required methods:

    densities(term, R, dR, ddR)   -> qⱼ as a Number (m = 1), Tuple or SVector
    combine(term, Q::SVector, b::BodyInfo) -> F(Q)

Optional methods (default: automatic differentiation with [`ad_backend`](@ref)):

    density_gradients(term, R, dR, ddR) -> ∂qⱼ/∂(R, R', R'') as an `m × 3`
        SMatrix, a 3-tuple (m = 1) or a tuple of 3-tuples
    combine_gradient(term, Q, b)       -> ∂F/∂Q as SVector/Tuple/Number
    applies_to(term, type)             -> whether the term acts on body type

All methods must be allocation-free so they can run inside GPU kernels.
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
    BodyInfo(R0, type)

Per-body data passed to [`combine`](@ref), e.g. to set a target area πR0².
"""
struct BodyInfo{T}
    R0::T
    type::Int32
end

function densities end
function combine end
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

density_gradients(term::BodyTerm, R, dR, ddR) = ad_density_gradients(ad_backend(term), term, R, dR, ddR)
combine_gradient(term::BodyTerm, Q, b) = ad_combine_gradient(ad_backend(term), term, Q, b)

function ad_density_gradients(::ForwardDiffAD, term, R, dR, ddR)
    f(v) = as_svector(densities(term, v[1], v[2], v[3]))
    return ForwardDiff.jacobian(f, SVector(R, dR, ddR))
end

ad_combine_gradient(::ForwardDiffAD, term, Q, b) = ForwardDiff.gradient(q -> combine(term, q, b), Q)

ad_density_gradients(::EnzymeAD, args...) = enzyme_missing()
ad_combine_gradient(::EnzymeAD, args...) = enzyme_missing()
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
the restriction applies to the body whose contour is evaluated (body i).
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

densities(t::OnTypes, R, dR, ddR) = densities(t.term, R, dR, ddR)
combine(t::OnTypes, Q, b) = combine(t.term, Q, b)
density_gradients(t::OnTypes, R, dR, ddR) = density_gradients(t.term, R, dR, ddR)
combine_gradient(t::OnTypes, Q, b) = combine_gradient(t.term, Q, b)
pressure(t::OnTypes, d, ti, tj, rel) = pressure(t.term, d, ti, tj, rel)
interaction_range(t::OnTypes) = interaction_range(t.term)
