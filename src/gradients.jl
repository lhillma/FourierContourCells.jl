"""
    Gradients

Work buffers for the energy gradient.

- `GR`, `GD`, `GDD`: `nθ × N` contour gradients ∂E/∂R, ∂E/∂R', ∂E/∂R'',
  views of the row blocks of `Gall` (`3nθ × N`, see [`Geometry`](@ref))
- `P`: `nθ × N` contact energy density at the contour nodes
- `gx`, `gy`, `gθ`: ∂E/∂x, ∂E/∂y, ∂E/∂θ per body
- `glab`: `ncoef × N` gradient with respect to lab-frame coefficients
- `gcoef`: `ncoef × N` gradient with respect to body-frame coefficients
  (row 1, a₀, already folded into aₙ, bₙ by the hard area constraint)
- `Ebody`: single-body energy per body
"""
struct Gradients{V <: AbstractVector, Mat <: AbstractMatrix, VMat <: AbstractMatrix}
    Gall::Mat
    GR::VMat
    GD::VMat
    GDD::VMat
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
    Gall = alloc(3nθ, N)
    return Gradients(Gall, row_blocks(Gall, nθ)..., alloc(nθ, N), alloc(N), alloc(N), alloc(N),
        alloc(nc, N), alloc(nc, N), alloc(N))
end

# Zero the accumulated gradients (P, glab and gcoef are overwritten, not
# accumulated, by their kernels). One work item per entry of Gall; the first N
# also zero the per-body vectors.
#
#   Gall          (out) 3nθ × N contour gradients
#   gx, gy, gθ, E (out) N, centre gradients and body energies
#   status        run status; nothing is done while it is frozen (see isactive)
@kernel function reset_kernel!(Gall, gx, gy, gθ, E, status)
    @inbounds if isactive(status)
        n = @index(Global)
        Gall[n] = zero(eltype(Gall))
        if n <= length(gx)
            gx[n] = zero(eltype(gx))
            gy[n] = zero(eltype(gy))
            gθ[n] = zero(eltype(gθ))
            E[n] = zero(eltype(E))
        end
    end
end

function reset!(g::Gradients; status = nothing)
    reset_kernel!(get_backend(g.Gall), WORKGROUP)(g.Gall, g.gx, g.gy, g.gθ, g.Ebody, status;
        ndrange = length(g.Gall))
    return g
end

## Single-body terms

# Accumulate contour gradients of all body terms for body i and return their
# energy. Recursion over the term tuple is unrolled at compile time.
Base.@propagate_inbounds _body_terms!(::Tuple{}, GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ) = zero(eltype(GR))
Base.@propagate_inbounds function _body_terms!(terms::Tuple, GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ)
    e = _body_term!(first(terms), GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ)
    return e + _body_terms!(Base.tail(terms), GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ)
end

# One body term for body i. With E = F(Q), Qⱼ = Σₖ qⱼ(Rₖ, R'ₖ, R''ₖ) Δφ
# (trapezoidal rule = spectral accuracy for periodic integrands), the chain rule
# gives the contour gradient at node k
#   ∂E/∂Rₖ = Σⱼ (∂F/∂Qⱼ) (∂qⱼ/∂R)(φₖ) Δφ,  and likewise for R'ₖ and R''ₖ.
# The first pass over the nodes computes Q, the second accumulates
#   (GR, GD, GDD)[k, i] += Jₖᵀ w,  Jₖ = ∂q/∂(R, R', R'') (m × 3),  w = ∂F/∂Q · Δφ.
Base.@propagate_inbounds function _body_term!(term, GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ)
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

# Single-body terms. One work item per body i. The body-term kernels run
# first in compute_gradients! and initialise the accumulated gradients (this
# replaces reset!): the contour gradients of body i are zeroed and then
# accumulated over the terms, E[i] is assigned and gx, gy, gθ are zeroed.
#
#   GR, GD, GDD  (out) nθ × N contour gradients ∂E/∂R, ∂E/∂R', ∂E/∂R''
#   E            (out) N, energy per body
#   gx, gy, gθ   (out) N, zeroed for the contact and bond kernels
#   R, dR, ddR   nθ × N contour radius and its φ-derivatives at the nodes
#   R0, type     N, radius scale and type (passed to `energy` via BodyInfo)
#   terms        tuple of BodyTerms
#   nθ, Δφ       number of nodes and node spacing 2π/nθ
#   status       run status; nothing is done while it is frozen (see isactive)
@kernel function body_terms_kernel!(GR, GD, GDD, E, gx, gy, gθ, @Const(R), @Const(dR), @Const(ddR),
        @Const(R0), @Const(type), terms, nθ, Δφ, status)
    @inbounds if isactive(status)
        i = @index(Global)
        for k in 1:nθ
            GR[k, i] = zero(eltype(GR))
            GD[k, i] = zero(eltype(GD))
            GDD[k, i] = zero(eltype(GDD))
        end
        gx[i] = zero(eltype(gx))
        gy[i] = zero(eltype(gy))
        gθ[i] = zero(eltype(gθ))
        info = BodyInfo(R0[i], type[i])
        E[i] = _body_terms!(terms, GR, GD, GDD, R, dR, ddR, i, info, nθ, Δφ)
    end
end

# Integrands of all terms at one node, concatenated into one SVector;
# `counts` holds Val(number of integrands) per term. Terms that do not apply
# to the body type contribute zeros without being evaluated.
@inline _node_integrands(::Tuple{}, ::Tuple{}, type, R, dR, ddR) = SVector{0, typeof(R)}()
@inline function _node_integrands(terms::Tuple, counts::Tuple, type, R, dR, ddR)
    t = first(terms)
    Q = SVector{_val(first(counts)), typeof(R)}
    q = applies_to(t, type) ? Q(as_svector(contour_integrands(t, R, dR, ddR))) : zero(Q)
    return vcat(q, _node_integrands(Base.tail(terms), Base.tail(counts), type, R, dR, ddR))
end

# Split Q = [Q_term; Q_rest] after the first m entries.
@inline _qhead(Q::SVector{L, T}, ::Val{m}) where {L, T, m} = SVector{m, T}(ntuple(c -> Q[c], Val(m)))
@inline _qtail(Q::SVector{L, T}, ::Val{m}) where {L, T, m} = SVector{L - m, T}(ntuple(c -> Q[m + c], Val(L - m)))

# Energy and contour gradient (∂E/∂R, ∂E/∂R', ∂E/∂R'') at one node of all
# terms, given the concatenated integrals Q (same chain rule as _body_term!)
# and the integrand counts of the terms; terms that do not apply to the body
# type are skipped.
@inline _node_gradients(::Tuple{}, ::Tuple{}, Q, info, R, dR, ddR, Δφ) = (zero(R), zero(SVector{3, typeof(R)}))
@inline function _node_gradients(terms::Tuple, counts::Tuple, Q, info, R, dR, ddR, Δφ)
    t = first(terms)
    m = first(counts)
    e, g = _node_gradients(Base.tail(terms), Base.tail(counts), _qtail(Q, m), info, R, dR, ddR, Δφ)
    applies_to(t, info.type) || return e, g
    J = as_jacobian(contour_integrand_gradients(t, R, dR, ddR))
    Qt = _qhead(Q, m)
    w = as_svector(energy_gradient(t, Qt, info)) * Δφ
    return e + energy(t, Qt, info), g + J' * w
end

"""
Number of contour integrals of a body term, from the inferred return type of
`contour_integrands` (a number, tuple or `SVector`), so the term is not
evaluated outside its domain. Falls back to evaluating it on the unit circle
if the type cannot be inferred.
"""
function n_integrands(term, ::Type{T}) where {T}
    m = _integrand_count(Base.promote_op(contour_integrands, typeof(term), T, T, T))
    return m === nothing ? length(as_svector(contour_integrands(term, one(T), zero(T), zero(T)))) : m
end
_integrand_count(::Type{<:Number}) = 1
_integrand_count(::Type{<:StaticVector{m}}) where {m} = m
_integrand_count(::Type{<:NTuple{m, Any}}) where {m} = m
_integrand_count(::Type) = nothing
_integrand_count(::Type{Union{}}) = nothing  # inference: always throws

# Node-parallel variant of body_terms_kernel! (see KernelLayout). One work item
# per (node k, body i), in workgroups of nθ × B bodies. Each work item stores
# its integrands qⱼ(φₖ) in local memory; after the barrier every work item of
# body i sums them to the same Qⱼ = Σₖ qⱼ(φₖ) Δφ and adds its own node's
#   (GR, GD, GDD)[k, i] += Jₖᵀ w,  w = ∂F/∂Q · Δφ.
# The kernel initialises the gradients like body_terms_kernel!: each node
# assigns its contour gradients, node k = 1 assigns the energy and zeroes
# gx, gy, gθ.
#
#   GR, GD, GDD  (out) nθ × N contour gradients
#   E            (out) N, energy per body
#   gx, gy, gθ   (out) N, zeroed for the contact and bond kernels
#   R, dR, ddR   nθ × N contour radius and its φ-derivatives at the nodes
#   R0, type     N, radius scale and type
#   terms        tuple of BodyTerms
#   counts       tuple of Val(number of integrands), one per term
#   N            number of bodies (the ndrange is padded, see node_launch)
#   Δφ           node spacing 2π/nθ
#   nq           Val(total number of integrands), sizes the local buffer
#   status       run status; nothing is done while it is frozen (uniform per launch)
@kernel function body_terms_node_kernel!(GR, GD, GDD, E, gx, gy, gθ, @Const(R), @Const(dR), @Const(ddR),
        @Const(R0), @Const(type), terms, counts, N, Δφ, nq, status)
    @uniform nθ = @groupsize()[1]
    @uniform B = @groupsize()[2]
    qbuf = @localmem eltype(GR) (nθ, B, _val(nq))
    k, i = @index(Global, NTuple)
    _, b = @index(Local, NTuple)
    @inbounds if i <= N && isactive(status)
        q = _node_integrands(terms, counts, type[i], R[k, i], dR[k, i], ddR[k, i])
        for c in 1:_val(nq)
            qbuf[k, b, c] = q[c]
        end
    end
    @synchronize
    k, i = @index(Global, NTuple)
    _, b = @index(Local, NTuple)
    # no @inbounds here: measured slower with it on CUDA (218 vs 191 µs, N = 16384)
    if i <= N && isactive(status)
        Q = _sum_nodes(qbuf, b, nθ, nq) * Δφ
        info = BodyInfo(R0[i], type[i])
        e, g = _node_gradients(terms, counts, Q, info, R[k, i], dR[k, i], ddR[k, i], Δφ)
        GR[k, i] = g[1]
        GD[k, i] = g[2]
        GDD[k, i] = g[3]
        if k == 1
            E[i] = e
            gx[i] = zero(eltype(gx))
            gy[i] = zero(eltype(gy))
            gθ[i] = zero(eltype(gθ))
        end
    end
end

_val(::Val{n}) where {n} = n

# SVector of the node sums Σₖ buf[k, b, c] for c = 1:nq.
@inline function _sum_nodes(buf, b, nθ, ::Val{nq}) where {nq}
    return SVector(ntuple(Val(nq)) do c
        s = zero(eltype(buf))
        for k in 1:nθ
            s += @inbounds buf[k, b, c]
        end
        s
    end)
end

function body_terms!(g::Gradients, terms::Tuple, geom::Geometry, bodies::Bodies, basis::FourierBasis,
        layout::KernelLayout = BodyParallel(); status = nothing)
    # the body-term kernels initialise the gradients; without terms, reset them
    isempty(terms) && return reset!(g; status)
    backend = get_backend(g.GR)
    N = nbodies(bodies)
    if uses_node_layout(layout, basis.nθ)
        T = eltype(g.GR)
        counts = map(t -> n_integrands(t, T), terms)
        wg, nd = node_launch(basis.nθ, N)
        body_terms_node_kernel!(backend, wg)(g.GR, g.GD, g.GDD, g.Ebody, g.gx, g.gy, g.gθ, geom.R, geom.dR,
            geom.ddR, bodies.R0, bodies.type, terms, map(Val, counts), Int32(N), basis.Δφ,
            Val(sum(counts)), status; ndrange = nd)
    else
        body_terms_kernel!(backend, WORKGROUP)(g.GR, g.GD, g.GDD, g.Ebody, g.gx, g.gy, g.gθ, geom.R, geom.dR,
            geom.ddR, bodies.R0, bodies.type, terms, basis.nθ, basis.Δφ, status; ndrange = N)
    end
    return g
end

## Contact pressure → contour and centre gradients

# Convert the contact energy density P = g on the contour into gradients of
# body i's contact energy Eᵢ = ∫_{Ωᵢ} g dA. One work item per body i. For
# g = H(d) this reproduces the force, torque and shape terms of Saito &
# Ishihara (2024), Eqs. (4) and (5), doi:10.1126/sciadv.adi8433.
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
#   cs          2 × nθ, (cos φₖ, sin φₖ) of the nodes
#   status      run status; nothing is done while it is frozen (see isactive)
@kernel function contact_gradient_kernel!(GR, gx, gy, gθ, @Const(P), @Const(R), @Const(dR), nθ, Δφ, @Const(cs), status)
    @inbounds if isactive(status)
        i = @index(Global)
        fx = zero(eltype(gx))
        fy = zero(eltype(gy))
        tθ = zero(eltype(gθ))
        for k in 1:nθ
            p = P[k, i] * Δφ
            r = R[k, i]
            r′ = dR[k, i]
            c = cs[1, k]
            s = cs[2, k]
            GR[k, i] += p * r
            fx += p * (r * c + r′ * s)
            fy += p * (r * s - r′ * c)
            tθ -= p * r * r′
        end
        gx[i] += fx
        gy[i] += fy
        gθ[i] += tθ
    end
end

function contact_gradient!(g::Gradients, geom::Geometry, basis::FourierBasis; status = nothing)
    backend = get_backend(g.GR)
    contact_gradient_kernel!(backend, WORKGROUP)(g.GR, g.gx, g.gy, g.gθ, g.P, geom.R, geom.dR,
        basis.nθ, basis.Δφ, basis.cs, status; ndrange = size(g.P, 2))
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
#   glab   ncoef × N, ∂E/∂[a₀; αₙ; βₙ] / R₀ from `pullback!` (scaled by R0 here)
#   coef   ncoef × N, body-frame coefficients; θ, R0: N, orientations and radius scales
#   K      number of shape modes
#
# The θ-gradient is not taken from the lab-frame coefficients: single-body
# terms ∮ q(R, R', R'') dφ are rotation invariant, so their exact θ-gradient
# vanishes and the discrete one is pure quadrature aliasing. Contact terms add
# their θ-gradient directly in `contact_gradient_kernel!`.
@kernel function body_frame_kernel!(gcoef, @Const(glab), @Const(coef), @Const(θ), @Const(R0), K)
    @inbounds begin
        i = @index(Global)
        body_frame!(gcoef, glab, coef, θ[i], R0[i], i, K)
    end
end

# The work of body_frame_kernel! for body i with orientation θi and radius
# scale r0 (shared with the fused update kernel, see euler_maruyama_kernel!).
Base.@propagate_inbounds function body_frame!(gcoef, glab, coef, θi, r0, i, K)
    ga0 = r0 * glab[1, i]
    a0 = coef[1, i]
    for m in 1:K
        n = m + 1
        gα = r0 * glab[1 + m, i]
        gβ = r0 * glab[1 + K + m, i]
        s, c = sincos(n * θi)
        a = coef[1 + m, i]
        b = coef[1 + K + m, i]
        gcoef[1 + m, i] = gα * c + gβ * s - ga0 * a / (2a0)
        gcoef[1 + K + m, i] = -gα * s + gβ * c - ga0 * b / (2a0)
    end
    gcoef[1, i] = ga0
    return nothing
end

"""
    pullback!(g, geom, bodies, basis)

Map contour gradients `GR`, `GD`, `GDD` to gradients with respect to the
body-frame coefficients. Since R = R₀ C·lab (and R' = R₀ D·lab, R'' = R₀ DD·lab),
the lab-frame gradient is ∂E/∂lab = R₀ (Cᵀ GR + Dᵀ GD + DDᵀ GDD) = R₀ Sᵀ Gall:
one stacked transposed matrix product (three on the CPU, see
`update_geometry!`), then `body_frame_kernel!` scales by R₀ and rotates it
into the body frame.
"""
function pullback!(g::Gradients, geom::Geometry, bodies::Bodies, basis::FourierBasis; frame::Bool = true)
    backend = get_backend(g.Gall)
    if stacked_products(backend)
        matmul!(g.glab, transpose(basis.S), g.Gall)
    else
        mul!(g.glab, transpose(basis.C), g.GR)
        mul!(g.glab, transpose(basis.D), g.GD, true, true)
        mul!(g.glab, transpose(basis.DD), g.GDD, true, true)
    end
    frame && body_frame_kernel!(backend, WORKGROUP)(g.gcoef, g.glab, bodies.coef, bodies.θ,
        bodies.R0, nmodes(basis); ndrange = nbodies(bodies))
    return g
end
