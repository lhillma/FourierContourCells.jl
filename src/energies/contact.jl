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

# Contact energy density Σⱼ Σ_terms g(d) at node k of body i (see
# contact_pressure_kernel!), shared by both kernel layouts.
Base.@propagate_inbounds function _node_pressure(k, i, x, y, R, lab, rmax, R0, type, parent, nidx, ncount,
        terms, Lx, Ly, K, cs, range)
    c = cs[1, k]
    s = cs[2, k]
    ux = x[i] + R[k, i] * c
    uy = y[i] + R[k, i] * s
    ti = type[i]
    pi_ = parent[i]
    p = zero(eltype(R))
    for q in 1:ncount[i]
        j = nidx[q, i]
        dx = minimum_image(ux - x[j], Lx)
        dy = minimum_image(uy - y[j], Ly)
        dist2 = dx^2 + dy^2
        lim = rmax[j] + range
        dist2 > lim^2 && continue
        dist = sqrt(dist2)
        # direction of u - rⱼ (ψ = 0 if u sits exactly at rⱼ, like atan(0, 0))
        cψ = ifelse(dist > 0, dx / dist, one(dist))
        sψ = ifelse(dist > 0, dy / dist, zero(dist))
        d = contour_radius_dir(lab, j, R0[j], K, cψ, sψ) - dist
        p += _pressure(terms, d, ti, type[j], pi_ == parent[j])
    end
    return p
end

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
#   Lx, Ly     box size; K: number of shape modes
#   cs         2 × nθ, (cos φₖ, sin φₖ) of the nodes (FourierBasis.cs)
#   range      largest interaction range of the terms beyond contact
#   status     run status; nothing is done while it is frozen (see isactive)
@kernel function contact_pressure_kernel!(P, @Const(x), @Const(y), @Const(R), @Const(lab),
        @Const(rmax), @Const(R0), @Const(type), @Const(parent), @Const(nidx), @Const(ncount),
        terms, Lx, Ly, K, @Const(cs), range, status)
    @inbounds if isactive(status)
        k, i = @index(Global, NTuple)
        P[k, i] = _node_pressure(k, i, x, y, R, lab, rmax, R0, type, parent, nidx, ncount, terms,
            Lx, Ly, K, cs, range)
    end
end

"""
    contact_pressure!(P, terms, geom, bodies, neighbors, box, basis)

Sum of contact energy densities g at every contour node of every body.
"""
function contact_pressure!(P, terms::Tuple, geom::Geometry, bodies::Bodies, nl::NeighborList,
        box::PeriodicBox, basis::FourierBasis; status = nothing)
    T = eltype(P)
    backend = get_backend(P)
    contact_pressure_kernel!(backend, (basis.nθ, max(1, WORKGROUP ÷ basis.nθ)))(P, bodies.x,
        bodies.y, geom.R, geom.lab, geom.rmax, bodies.R0, bodies.type, bodies.parent, nl.idx,
        nl.count, terms, box.Lx, box.Ly, nmodes(basis), basis.cs, T(contact_range(terms)), status;
        ndrange = (basis.nθ, nbodies(bodies)))
    return P
end

# Node-parallel fusion of contact_pressure_kernel! and contact_gradient_kernel!
# (see KernelLayout). One work item per (node k, body i), in workgroups of
# nθ × B bodies. Each work item computes the energy density gₖ at its node,
# stores it in P and adds ∂E/∂Rₖ = gₖ Rₖ Δφ to GR, and puts its contributions
# to the centre force and torque (contact_gradient_kernel!) in local memory;
# after the barrier, node k = 1 sums them over the nodes of body i.
#
#   P, GR       (out, in/out) nθ × N, energy density and ∂E/∂R (accumulated)
#   gx, gy, gθ  (in/out) N, ∂E/∂x, ∂E/∂y, ∂E/∂θ (accumulated)
#   N           number of bodies (the ndrange is padded, see node_launch)
#   Δφ          node spacing (quadrature weight)
#   status      run status; nothing is done while it is frozen (uniform per launch)
#   other arguments as in contact_pressure_kernel!; dR: nθ × N, dR/dφ
@kernel function contact_node_kernel!(P, GR, gx, gy, gθ, @Const(x), @Const(y), @Const(R),
        @Const(dR), @Const(lab), @Const(rmax), @Const(R0), @Const(type), @Const(parent),
        @Const(nidx), @Const(ncount), terms, N, Lx, Ly, K, Δφ, @Const(cs), range, status)
    @uniform nθ = @groupsize()[1]
    @uniform B = @groupsize()[2]
    fbuf = @localmem eltype(GR) (nθ, B, 3)
    k, i = @index(Global, NTuple)
    _, b = @index(Local, NTuple)
    @inbounds if i <= N && isactive(status)
        g = _node_pressure(k, i, x, y, R, lab, rmax, R0, type, parent, nidx, ncount, terms,
            Lx, Ly, K, cs, range)
        P[k, i] = g
        p = g * Δφ
        r = R[k, i]
        r′ = dR[k, i]
        c = cs[1, k]
        s = cs[2, k]
        GR[k, i] += p * r
        fbuf[k, b, 1] = p * (r * c + r′ * s)
        fbuf[k, b, 2] = p * (r * s - r′ * c)
        fbuf[k, b, 3] = -p * r * r′
    end
    @synchronize
    k, i = @index(Global, NTuple)
    _, b = @index(Local, NTuple)
    @inbounds if i <= N && k == 1 && isactive(status)
        fx = zero(eltype(GR))
        fy = zero(eltype(GR))
        tθ = zero(eltype(GR))
        for kk in 1:nθ
            fx += fbuf[kk, b, 1]
            fy += fbuf[kk, b, 2]
            tθ += fbuf[kk, b, 3]
        end
        gx[i] += fx
        gy[i] += fy
        gθ[i] += tθ
    end
end

"""
    contact_gradients!(g, terms, geom, bodies, neighbors, box, basis)

Contact energy density `g.P` and its gradients: ∂E/∂R into `g.GR` and the
centre force and torque into `g.gx`, `g.gy`, `g.gθ` (accumulated). Uses the
fused node-parallel kernel on all backends unless nθ exceeds a workgroup.
"""
function contact_gradients!(g::Gradients, terms::Tuple, geom::Geometry, bodies::Bodies,
        nl::NeighborList, box::PeriodicBox, basis::FourierBasis; status = nothing)
    if uses_node_layout(NodeParallel(), basis.nθ)
        T = eltype(g.GR)
        N = nbodies(bodies)
        wg, nd = node_launch(basis.nθ, N)
        contact_node_kernel!(get_backend(g.GR), wg)(g.P, g.GR, g.gx, g.gy, g.gθ, bodies.x,
            bodies.y, geom.R, geom.dR, geom.lab, geom.rmax, bodies.R0, bodies.type,
            bodies.parent, nl.idx, nl.count, terms, Int32(N), box.Lx, box.Ly, nmodes(basis),
            basis.Δφ, basis.cs, T(contact_range(terms)), status; ndrange = nd)
    else
        contact_pressure!(g.P, terms, geom, bodies, nl, box, basis; status)
        contact_gradient!(g, geom, basis; status)
    end
    return g
end
