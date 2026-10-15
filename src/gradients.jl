"""
    Gradients

Work buffers for the energy gradient.

- `GR`, `GD`, `GDD`: `nθ × N` contour gradients ∂E/∂R, ∂E/∂R', ∂E/∂R''
- `P`: `nθ × N` contact energy density at the contour nodes
- `gx`, `gy`, `gθ`: ∂E/∂x, ∂E/∂y, ∂E/∂θ per body
- `glab`: `ncoef × N` gradient with respect to lab-frame coefficients
- `gcoef`: `ncoef × N` gradient with respect to body-frame coefficients
  (row 1, a₀, already folded into aₙ, bₙ by the hard area constraint)
- `Ebody`: single-body energy per body
"""
struct Gradients{V <: AbstractVector, Mat <: AbstractMatrix}
    GR::Mat
    GD::Mat
    GDD::Mat
    P::Mat
    gx::V
    gy::V
    gθ::V
    glab::Mat
    gcoef::Mat
    Ebody::V
end

function Gradients(basis::FourierBasis, bodies::Bodies)
    N = nbodies(bodies)
    T = eltype(bodies.x)
    backend = get_backend(bodies.x)
    alloc(dims...) = KernelAbstractions.zeros(backend, T, dims...)
    nθ, nc = basis.nθ, ncoef(basis)
    return Gradients(alloc(nθ, N), alloc(nθ, N), alloc(nθ, N), alloc(nθ, N), alloc(N),
        alloc(N), alloc(N), alloc(nc, N), alloc(nc, N), alloc(N))
end

function reset!(g::Gradients)
    for f in fieldnames(Gradients)
        fill!(getfield(g, f), 0)
    end
    return g
end

## Single-body terms

# Accumulate contour gradients of all body terms for body i and return their
# energy. Recursion over the term tuple is unrolled at compile time.
@inline _body_terms!(::Tuple{}, GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ) = zero(eltype(GR))
@inline function _body_terms!(terms::Tuple, GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ)
    e = _body_term!(first(terms), GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ)
    return e + _body_terms!(Base.tail(terms), GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ)
end

# One body term for body i. With E = F(Q), Qⱼ = Σₖ qⱼ(Rₖ, R'ₖ, R''ₖ) Δφ
# (trapezoidal rule = spectral accuracy for periodic integrands), the chain rule
# gives the contour gradient at node k
#   ∂E/∂Rₖ = Σⱼ (∂F/∂Qⱼ) (∂qⱼ/∂R)(φₖ) Δφ,  and likewise for R'ₖ and R''ₖ.
# The first pass over the nodes computes Q, the second accumulates
#   (GR, GD, GDD)[k, i] += Jₖᵀ w,  Jₖ = ∂q/∂(R, R', R'') (m × 3),  w = ∂F/∂Q · Δφ.
@inline function _body_term!(term, GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ)
    applies_to(term, info.type) || return zero(eltype(GR))
    Q = as_svector(contour_integrands(term, R[1, i], dR[1, i], ddR[1, i]))
    for k in 2:nθ
        Q += as_svector(contour_integrands(term, R[k, i], dR[k, i], ddR[k, i]))
    end
    Q *= Δφ
    w = as_svector(energy_gradient(term, Q, info)) * Δφ
    for k in 1:nθ
        J = as_jacobian(contour_integrand_gradients(term, R[k, i], dR[k, i], ddR[k, i]))
        g = J' * w
        GR[k, i] += g[1]
        GD[k, i] += g[2]
        GDD[k, i] += g[3]
    end
    return energy(term, Q, info)
end

# Single-body terms. One work item per body i.
#
#   GR, GD, GDD  (in/out) nθ × N contour gradients ∂E/∂R, ∂E/∂R', ∂E/∂R'' (accumulated)
#   E            (in/out) N, energy per body (accumulated)
#   R, dR, ddR   nθ × N contour radius and its φ-derivatives at the nodes
#   R0, type     N, radius scale and type (passed to `energy` via BodyInfo)
#   terms        tuple of BodyTerms
#   nθ, Δφ       number of nodes and node spacing 2π/nθ
@kernel function body_terms_kernel!(GR, GD, GDD, E, @Const(R), @Const(dR), @Const(ddR),
        @Const(R0), @Const(type), terms, nθ, Δφ)
    i = @index(Global)
    info = BodyInfo(R0[i], type[i])
    E[i] += _body_terms!(terms, GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ)
end

function body_terms!(g::Gradients, terms::Tuple, geom::Geometry, bodies::Bodies, basis::FourierBasis)
    isempty(terms) && return g
    backend = get_backend(g.GR)
    body_terms_kernel!(backend, WORKGROUP)(g.GR, g.GD, g.GDD, g.Ebody, geom.R, geom.dR,
        geom.ddR, bodies.R0, bodies.type, terms, basis.nθ, basis.Δφ; ndrange = nbodies(bodies))
    return g
end

## Contact pressure → contour and centre gradients

# Convert the contact energy density P = g on the contour into gradients of
# body i's contact energy Eᵢ = ∫_{Ωᵢ} g dA. One work item per body i.
#
# Moving the boundary of Ωᵢ changes Eᵢ by ∮ g δn ds, where δn ds is the area
# swept by the boundary element. The three kinds of motion:
#   - changing R(φₖ) by δR sweeps R δR dφ        ⇒ ∂E/∂Rₖ = gₖ Rₖ Δφ
#   - translating by δr sweeps (n ds)·δr, with the outward normal element
#       n ds = (R e_φ - R' e_⊥) dφ = (R cos φ + R' sin φ, R sin φ - R' cos φ) dφ
#     (e_φ = (cos φ, sin φ), e_⊥ = (-sin φ, cos φ); from rotating the tangent
#     dp/dφ = R' e_φ + R e_⊥ by -90°)        ⇒ ∂E/∂r = Σₖ gₖ (n ds)ₖ
#   - rotating by δθ changes R(φ) = R̃(φ - θ) by -R' δθ ⇒ ∂E/∂θ = -Σₖ gₖ Rₖ R'ₖ Δφ
#
#   GR          (in/out) nθ × N, ∂E/∂R (accumulated)
#   gx, gy, gθ  (in/out) N, ∂E/∂x, ∂E/∂y, ∂E/∂θ (accumulated)
#   P           nθ × N, contact energy density g at the nodes
#   R, dR       nθ × N, contour radius and dR/dφ
#   nθ, Δφ      number of nodes and node spacing
@kernel function contact_gradient_kernel!(GR, gx, gy, gθ, @Const(P), @Const(R), @Const(dR), nθ, Δφ)
    i = @index(Global)
    fx = zero(eltype(gx))
    fy = zero(eltype(gy))
    tθ = zero(eltype(gθ))
    for k in 1:nθ
        p = P[k, i] * Δφ
        r = R[k, i]
        r′ = dR[k, i]
        s, c = sincos((k - 1) * Δφ)
        GR[k, i] += p * r
        fx += p * (r * c + r′ * s)
        fy += p * (r * s - r′ * c)
        tθ -= p * r * r′
    end
    gx[i] += fx
    gy[i] += fy
    gθ[i] += tθ
end

function contact_gradient!(g::Gradients, geom::Geometry, basis::FourierBasis)
    backend = get_backend(g.GR)
    contact_gradient_kernel!(backend, WORKGROUP)(g.GR, g.gx, g.gy, g.gθ, g.P, geom.R, geom.dR,
        basis.nθ, basis.Δφ; ndrange = size(g.P, 2))
    return g
end

## Chain rule to the dynamical variables

# Map gradients with respect to lab-frame coefficients to body-frame ones.
# One work item per body i.
#
# Lab coefficients are (aₙ, bₙ) rotated by nθ (see lab_frame_kernel!):
#   αₙ = aₙ cos nθ - bₙ sin nθ,   βₙ = aₙ sin nθ + bₙ cos nθ,
# so by the chain rule the gradient is rotated back by the transposed rotation:
#   ∂E/∂aₙ =  ∂E/∂αₙ cos nθ + ∂E/∂βₙ sin nθ
#   ∂E/∂bₙ = -∂E/∂αₙ sin nθ + ∂E/∂βₙ cos nθ.
# The hard area constraint a₀ = √(1 - Σ(aₙ² + bₙ²)/2) makes a₀ depend on the
# modes, ∂a₀/∂aₙ = -aₙ/(2a₀), adding ∂E/∂a₀ · ∂a₀/∂aₙ (likewise for bₙ).
#
#   gcoef  (out) ncoef × N, ∂E/∂[a₀; aₙ; bₙ] (row 1 keeps the raw ∂E/∂a₀)
#   glab   ncoef × N, ∂E/∂[a₀; αₙ; βₙ] from `pullback!`
#   coef   ncoef × N, body-frame coefficients; θ: N, orientations
#   K      number of shape modes
#
# The θ-gradient is not taken from the lab-frame coefficients: single-body
# terms ∮ q(R, R', R'') dφ are rotation invariant, so their exact θ-gradient
# vanishes and the discrete one is pure quadrature aliasing. Contact terms add
# their θ-gradient directly in `contact_gradient_kernel!`.
@kernel function body_frame_kernel!(gcoef, @Const(glab), @Const(coef), @Const(θ), K)
    i = @index(Global)
    ga0 = glab[1, i]
    a0 = coef[1, i]
    for m in 1:K
        n = m + 1
        gα = glab[1 + m, i]
        gβ = glab[1 + K + m, i]
        s, c = sincos(n * θ[i])
        a = coef[1 + m, i]
        b = coef[1 + K + m, i]
        gcoef[1 + m, i] = gα * c + gβ * s - ga0 * a / (2a0)
        gcoef[1 + K + m, i] = -gα * s + gβ * c - ga0 * b / (2a0)
    end
    gcoef[1, i] = ga0
end

"""
    pullback!(g, geom, bodies, basis)

Map contour gradients `GR`, `GD`, `GDD` to gradients with respect to the
body-frame coefficients. Since R = R₀ C·lab (and R' = R₀ D·lab, R'' = R₀ DD·lab),
the lab-frame gradient is ∂E/∂lab = R₀ (Cᵀ GR + Dᵀ GD + DDᵀ GDD): three
transposed matrix products, then `body_frame_kernel!` rotates it into the body
frame.
"""
function pullback!(g::Gradients, geom::Geometry, bodies::Bodies, basis::FourierBasis)
    mul!(g.glab, transpose(basis.C), g.GR)
    mul!(g.glab, transpose(basis.D), g.GD, true, true)
    mul!(g.glab, transpose(basis.DD), g.GDD, true, true)
    g.glab .*= reshape(bodies.R0, 1, :)
    backend = get_backend(g.GR)
    body_frame_kernel!(backend, WORKGROUP)(g.gcoef, g.glab, bodies.coef, bodies.θ,
        nmodes(basis); ndrange = nbodies(bodies))
    return g
end
