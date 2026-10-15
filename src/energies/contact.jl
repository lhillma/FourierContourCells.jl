"""
    Overlap(J; related = J)

Excluded-volume interaction of Saito & Ishihara (2024): the energy is the
overlap area weighted by a coupling J, i.e. g(d) = J·H(d). Couplings can be
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

@inline _pressure(::Tuple{}, d, ti, tj, rel) = zero(d)
@inline function _pressure(terms::Tuple, d, ti, tj, rel)
    t = first(terms)
    g = applies_to(t, ti) ? pressure(t, d, ti, tj, rel) : zero(d)
    return g + _pressure(Base.tail(terms), d, ti, tj, rel)
end

contact_range(terms::Tuple) = isempty(terms) ? 0.0 : maximum(interaction_range, terms)

@kernel function contact_pressure_kernel!(P, @Const(x), @Const(y), @Const(R), @Const(lab),
        @Const(rmax), @Const(R0), @Const(type), @Const(parent), @Const(perm), @Const(start),
        @Const(stop), terms, Lx, Ly, ncx, ncy, wx, wy, K, Δφ, range)
    k, i = @index(Global, NTuple)
    s, c = sincos((k - 1) * Δφ)
    xi = x[i]
    yi = y[i]
    ux = xi + R[k, i] * c
    uy = yi + R[k, i] * s
    ti = type[i]
    pi_ = parent[i]
    cx, cy = grid_coords(ncx, ncy, wx, wy, xi, yi)
    rx = ifelse(ncx >= Int32(3), Int32(1), Int32(0))
    ry = ifelse(ncy >= Int32(3), Int32(1), Int32(0))
    p = zero(eltype(P))
    for oy in (-ry):ry, ox in (-rx):rx
        nx = mod(cx + ox, ncx)
        ny = mod(cy + oy, ncy)
        cell = nx + ncx * ny + Int32(1)
        for q in start[cell]:stop[cell]
            j = perm[q]
            j == i && continue
            dx = minimum_image(ux - x[j], Lx)
            dy = minimum_image(uy - y[j], Ly)
            dist2 = dx^2 + dy^2
            lim = rmax[j] + range
            dist2 > lim^2 && continue
            dist = sqrt(dist2)
            d = contour_radius(lab, j, R0[j], K, atan(dy, dx)) - dist
            p += _pressure(terms, d, ti, type[j], pi_ == parent[j])
        end
    end
    P[k, i] = p
end

"""
    contact_pressure!(P, terms, geom, bodies, cells, box, basis)

Sum of contact energy densities g at every contour node of every body.
"""
function contact_pressure!(P, terms::Tuple, geom::Geometry, bodies::Bodies, cl::CellList,
        box::PeriodicBox, basis::FourierBasis)
    T = eltype(P)
    backend = get_backend(P)
    contact_pressure_kernel!(backend, (basis.nθ, max(1, WORKGROUP ÷ basis.nθ)))(P, bodies.x,
        bodies.y, geom.R, geom.lab, geom.rmax, bodies.R0, bodies.type, bodies.parent, cl.perm,
        cl.start, cl.stop, terms, box.Lx, box.Ly, cl.ncx, cl.ncy, cl.wx, cl.wy, nmodes(basis),
        basis.Δφ, T(contact_range(terms)); ndrange = (basis.nθ, nbodies(bodies)))
    return P
end
