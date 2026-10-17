# Bonded interactions between body centres (an extension of the model of
# Saito & Ishihara (2024), doi:10.1126/sciadv.adi8433, e.g. to anchor a
# nucleus to its cell).

"""
    HarmonicBond(k; r0 = 0)

Harmonic spring E = (k/2) (|rᵢ - rⱼ| - r0)² between bonded centres, e.g. a
crude cytoskeleton anchoring a nucleus to the centre of its cell (`r0 = 0`).
Derivatives are implemented by hand.
"""
struct HarmonicBond{T} <: BondTerm
    k::T
    r0::T
end
HarmonicBond(k; r0 = zero(k)) = HarmonicBond(promote(k, r0)...)

@inline function bond_energy(b::HarmonicBond, Δ, ti, tj)
    r = sqrt(Δ[1]^2 + Δ[2]^2)
    return b.k / 2 * (r - b.r0)^2
end

# ∂E/∂Δ = k (1 - r0/r) Δ; at r = 0 the gradient k Δ = 0 (r0 = 0) or is
# undefined (r0 > 0, direction-less), and zero is returned
@inline function bond_gradient(b::HarmonicBond, Δ, ti, tj)
    r = sqrt(Δ[1]^2 + Δ[2]^2)
    f = ifelse(r > 0, b.k * (1 - b.r0 / r), ifelse(iszero(b.r0), oftype(r, b.k), zero(r)))
    return f * Δ
end

"""
    parent_bonds(bodies) -> Vector{Tuple{Int, Int}}

One bond between every body and its parent (bodies that are their own parent
get none), e.g. to anchor every nucleus to its cell.
"""
parent_bonds(b::Bodies) = [(i, Int(p)) for (i, p) in enumerate(to_host(b.parent)) if p != i]

"""
    BondList(bonds, N; backend)

Per-body list of bond partners, as a padded `capacity × N` matrix `idx` with
`count[i]` partners of body i in column i. Each bond (i, j) appears in the
columns of both bodies, so every body gathers its own bond forces (no
atomics). Built once on the host, as bonds are fixed.
"""
struct BondList{MI <: AbstractMatrix{Int32}, VI <: AbstractVector{Int32}}
    idx::MI
    count::VI
end

function BondList(bonds, N::Integer; backend = CPU())
    count = zeros(Int32, N)
    seen = Set{Tuple{Int, Int}}()
    for (i, j) in bonds
        (1 <= i <= N && 1 <= j <= N) || throw(ArgumentError("bond ($i, $j) refers to a body outside 1:$N"))
        i == j && throw(ArgumentError("bond ($i, $j) connects a body to itself"))
        key = minmax(i, j)
        key in seen && throw(ArgumentError("duplicate bond ($i, $j)"))
        push!(seen, key)
        count[i] += 1
        count[j] += 1
    end
    idx = zeros(Int32, max(1, maximum(count; init = 0)), N)
    fill!(count, 0)
    for (i, j) in bonds, (a, b) in ((i, j), (j, i))
        count[a] += 1
        idx[count[a], a] = b
    end
    return BondList(to_backend(backend, idx), to_backend(backend, count))
end

nbonds(bl::BondList) = Int(sum(to_host(bl.count))) ÷ 2

# Sum of ∂E/∂Δ over a tuple of bond terms (unrolled at compile time), skipping
# terms that do not apply to the pair of types.
@inline _bond_gradient(::Tuple{}, Δ, ti, tj) = zero(Δ)
@inline function _bond_gradient(terms::Tuple, Δ, ti, tj)
    t = first(terms)
    g = applies_to(t, ti) && applies_to(t, tj) ? as_svector2(bond_gradient(t, Δ, ti, tj)) : zero(Δ)
    return g + _bond_gradient(Base.tail(terms), Δ, ti, tj)
end

@inline as_svector2(g::SVector{2}) = g
@inline as_svector2(g::Tuple) = SVector{2}(g)

# Bond forces on body centres. One work item per body i: for every bond
# partner j, Δ = rᵢ - rⱼ (minimum image) and ∂E/∂rᵢ = ∂E/∂Δ (since
# ∂Δ/∂rᵢ = 1). The partner gets ∂E/∂rⱼ = -∂E/∂Δ from its own work item,
# which evaluates the gradient at -Δ with swapped types (equal by the
# required symmetry of the bond energy).
#
#   gx, gy     (in/out) N, ∂E/∂x, ∂E/∂y (accumulated)
#   x, y       N, wrapped body positions;  type: N, body types
#   bidx, bcount  bond partners of each body (see BondList)
#   terms      tuple of BondTerms;  Lx, Ly: box size
@kernel function bond_kernel!(gx, gy, @Const(x), @Const(y), @Const(type), @Const(bidx),
        @Const(bcount), terms, Lx, Ly)
    @inbounds begin
        i = @index(Global)
        gi = SVector(zero(eltype(gx)), zero(eltype(gx)))
        for q in 1:bcount[i]
            j = bidx[q, i]
            Δ = SVector(minimum_image(x[i] - x[j], Lx), minimum_image(y[i] - y[j], Ly))
            gi += _bond_gradient(terms, Δ, type[i], type[j])
        end
        gx[i] += gi[1]
        gy[i] += gi[2]
    end
end

"Add the bond forces of `terms` on all bonds in `bl` to the centre gradients."
function bond_gradient!(g::Gradients, terms::Tuple, bodies::Bodies, bl::BondList, box::PeriodicBox)
    (isempty(terms) || nbodies(bodies) == 0) && return g
    bond_kernel!(get_backend(g.gx), WORKGROUP)(g.gx, g.gy, bodies.x, bodies.y, bodies.type, bl.idx,
        bl.count, terms, box.Lx, box.Ly; ndrange = nbodies(bodies))
    return g
end
