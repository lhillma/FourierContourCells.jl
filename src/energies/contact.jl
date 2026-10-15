"""
    Overlap(J; related = J)

Excluded-volume interaction of Saito & Ishihara (2024),
doi:10.1126/sciadv.adi8433: the energy is the overlap area weighted by a
coupling J, i.e. g(d) = J·H(d). The coupling tables are an extension. Couplings can be
scalars (one body type) or `ntypes × ntypes` matrices indexed by
`(type_i, type_j)`, like the J table of the cellular Potts model. `related`
applies to pairs of bodies with the same parent (e.g. a cell and its nucleus);
a negative coupling there makes a body prefer to lie inside the other.
"""
struct Overlap{NT, T, L} <: ContactTerm
    J::SMatrix{NT, NT, T, L}
    Jrelated::SMatrix{NT, NT, T, L}
end

function Overlap(J::AbstractMatrix; related::AbstractMatrix = J)
    NT = size(J, 1)
    size(J) == size(related) == (NT, NT) || throw(DimensionMismatch("coupling tables must be square and equal-sized"))
    T = promote_type(eltype(J), eltype(related))
    return Overlap(SMatrix{NT, NT, T}(J), SMatrix{NT, NT, T}(related))
end
Overlap(J::Real; related::Real = J) = Overlap(fill(J, 1, 1); related = fill(related, 1, 1))

@inline function pressure(o::Overlap, d, ti, tj, related)
    J = related ? o.Jrelated[ti, tj] : o.J[ti, tj]
    return ifelse(d >= 0, oftype(d, J), zero(d))
end

"""
    LinearRepulsion(k)

Soft-core repulsion with energy density g(d) = k·d for d > 0, i.e. a penalty
growing linearly with the penetration depth (a 'harder' core than
[`Overlap`](@ref)).
"""
struct LinearRepulsion{T} <: ContactTerm
    k::T
end

@inline pressure(r::LinearRepulsion, d, ti, tj, related) = ifelse(d > 0, oftype(d, r.k) * d, zero(d))

# Sum of g(d) over a tuple of contact terms, skipping terms that do not apply
# to the type of body i. Recursion over the tuple is unrolled at compile time,
# so heterogeneous term tuples are type-stable inside kernels.
@inline _pressure(::Tuple{}, d, ti, tj, rel) = zero(d)
@inline function _pressure(terms::Tuple, d, ti, tj, rel)
    t = first(terms)
    g = applies_to(t, ti) ? pressure(t, d, ti, tj, rel) : zero(d)
    return g + _pressure(Base.tail(terms), d, ti, tj, rel)
end

"Largest interaction range beyond contact of a tuple of contact terms."
contact_range(terms::Tuple) = isempty(terms) ? 0.0 : maximum(interaction_range, terms)

# Contact energy density at every contour node. The sharp-interface contour
# formulation and the inside test Rⱼ(arg(u - rⱼ)) > |u - rⱼ| (with the rmax
# pre-check) follow Eqs. (4) and (6) of Saito & Ishihara (2024),
# doi:10.1126/sciadv.adi8433. One work item per (node k,
# body i): the node sits at u = rᵢ + Rᵢ(φₖ) (cos φₖ, sin φₖ). For each listed
# neighbour j, the vector u - rⱼ (minimum image) has length dist and angle ψ;
# the penetration depth of u into j is d = Rⱼ(ψ) - dist (d > 0: u inside j).
# All contact terms are evaluated at d and summed. Nodes farther than
# rmaxⱼ + range from rⱼ are skipped without evaluating Rⱼ(ψ).
#
#   P          (out) nθ × N, Σⱼ Σ_terms g(d) at node k of body i
#   x, y       N, wrapped body positions
#   R          nθ × N, contour radii at the nodes
#   lab        ncoef × N, lab-frame coefficients (to evaluate Rⱼ(ψ) off-grid)
#   rmax, R0   N, contour radius bound and radius scale per body
#   type       N, body types; parent: N, parent ids (related = same parent)
#   nidx, ncount  neighbour list (see NeighborList)
#   terms      tuple of ContactTerms
#   Lx, Ly     box size; K: number of shape modes; Δφ: node spacing
#   range      largest interaction range of the terms beyond contact
@kernel function contact_pressure_kernel!(P, @Const(x), @Const(y), @Const(R), @Const(lab),
        @Const(rmax), @Const(R0), @Const(type), @Const(parent), @Const(nidx), @Const(ncount),
        terms, Lx, Ly, K, Δφ, range)
    k, i = @index(Global, NTuple)
    s, c = sincos((k - 1) * Δφ)
    ux = x[i] + R[k, i] * c
    uy = y[i] + R[k, i] * s
    ti = type[i]
    pi_ = parent[i]
    p = zero(eltype(P))
    for q in 1:ncount[i]
        j = nidx[q, i]
        dx = minimum_image(ux - x[j], Lx)
        dy = minimum_image(uy - y[j], Ly)
        dist2 = dx^2 + dy^2
        lim = rmax[j] + range
        dist2 > lim^2 && continue
        dist = sqrt(dist2)
        d = contour_radius(lab, j, R0[j], K, atan(dy, dx)) - dist
        p += _pressure(terms, d, ti, type[j], pi_ == parent[j])
    end
    P[k, i] = p
end

"""
    contact_pressure!(P, terms, geom, bodies, neighbors, box, basis)

Sum of contact energy densities g at every contour node of every body.
"""
function contact_pressure!(P, terms::Tuple, geom::Geometry, bodies::Bodies, nl::NeighborList,
        box::PeriodicBox, basis::FourierBasis)
    T = eltype(P)
    backend = get_backend(P)
    contact_pressure_kernel!(backend, (basis.nθ, max(1, WORKGROUP ÷ basis.nθ)))(P, bodies.x,
        bodies.y, geom.R, geom.lab, geom.rmax, bodies.R0, bodies.type, bodies.parent, nl.idx,
        nl.count, terms, box.Lx, box.Ly, nmodes(basis), basis.Δφ, T(contact_range(terms));
        ndrange = (basis.nθ, nbodies(bodies)))
    return P
end
