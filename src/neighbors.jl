"""
    CellList(box, cutoff, N; backend)

Uniform grid for neighbour search. Bodies are sorted by grid cell index, and
`start[c]:stop[c]` indexes the sorted permutation `perm` for cell `c`. Grid
cells are at least `cutoff` wide, so interacting pairs live in adjacent cells.
A dimension with fewer than three grid cells collapses to a single cell, in
which case all bodies along it are checked with the minimum image convention.
"""
struct CellList{T, VI <: AbstractVector{Int32}}
    ncx::Int32
    ncy::Int32
    wx::T
    wy::T
    cellid::VI
    perm::VI
    start::VI
    stop::VI
end

function CellList(box::PeriodicBox{T}, cutoff, N::Integer; backend = CPU()) where {T}
    ncells(L) = (n = floor(Int, L / cutoff); n >= 3 ? n : 1)
    ncx, ncy = ncells(box.Lx), ncells(box.Ly)
    alloc(n) = KernelAbstractions.zeros(backend, Int32, n)
    return CellList{T, typeof(alloc(0))}(ncx, ncy, T(box.Lx / ncx), T(box.Ly / ncy),
        alloc(N), alloc(N), alloc(ncx * ncy), alloc(ncx * ncy))
end

"Grid cell coordinates (0-based) of a wrapped position."
@inline grid_coords(cl_ncx, cl_ncy, wx, wy, x, y) =
    (min(unsafe_trunc(Int32, x / wx), cl_ncx - Int32(1)),
     min(unsafe_trunc(Int32, y / wy), cl_ncy - Int32(1)))

@kernel function cell_index_kernel!(cellid, @Const(x), @Const(y), ncx, ncy, wx, wy)
    i = @index(Global)
    cx, cy = grid_coords(ncx, ncy, wx, wy, x[i], y[i])
    cellid[i] = cx + ncx * cy + Int32(1)
end

@kernel function cell_bounds_kernel!(start, stop, @Const(cellid), @Const(perm), N)
    s = @index(Global)
    c = cellid[perm[s]]
    if s == 1 || cellid[perm[s - 1]] != c
        start[c] = s
    end
    if s == N || cellid[perm[s + 1]] != c
        stop[c] = s
    end
end

"Rebuild the cell list from the current (wrapped) body positions."
function update_celllist!(cl::CellList, bodies::Bodies)
    backend = get_backend(bodies.x)
    N = nbodies(bodies)
    cell_index_kernel!(backend, WORKGROUP)(cl.cellid, bodies.x, bodies.y, cl.ncx, cl.ncy,
        cl.wx, cl.wy; ndrange = N)
    sortperm!(cl.perm, cl.cellid)
    fill!(cl.start, 1)
    fill!(cl.stop, 0)
    cell_bounds_kernel!(backend, WORKGROUP)(cl.start, cl.stop, cl.cellid, cl.perm, Int32(N);
        ndrange = N)
    return cl
end

"""
    NeighborList(N; capacity = 32, backend)

Per-body list of potentially interacting bodies, stored as a padded
`capacity × N` matrix `idx` with `count[i]` valid entries in column `i`.
Body j is listed for body i if their centres are closer than
rmax_i + rmax_j + range, which no contour point of i can reach otherwise.
The capacity grows automatically when exceeded.
"""
mutable struct NeighborList{MI <: AbstractMatrix{Int32}, VI <: AbstractVector{Int32}}
    idx::MI
    count::VI
end

function NeighborList(N::Integer; capacity::Integer = 32, backend = CPU())
    return NeighborList(KernelAbstractions.zeros(backend, Int32, capacity, N),
        KernelAbstractions.zeros(backend, Int32, N))
end

capacity(nl::NeighborList) = size(nl.idx, 1)

@kernel function neighbor_kernel!(idx, count, @Const(x), @Const(y), @Const(rmax), @Const(perm),
        @Const(start), @Const(stop), Lx, Ly, ncx, ncy, wx, wy, range, cap)
    i = @index(Global)
    xi = x[i]
    yi = y[i]
    ri = rmax[i] + range
    cx, cy = grid_coords(ncx, ncy, wx, wy, xi, yi)
    rx = ifelse(ncx >= Int32(3), Int32(1), Int32(0))
    ry = ifelse(ncy >= Int32(3), Int32(1), Int32(0))
    n = Int32(0)
    for oy in (-ry):ry, ox in (-rx):rx
        cell = mod(cx + ox, ncx) + ncx * mod(cy + oy, ncy) + Int32(1)
        for q in start[cell]:stop[cell]
            j = perm[q]
            j == i && continue
            dx = minimum_image(xi - x[j], Lx)
            dy = minimum_image(yi - y[j], Ly)
            if dx^2 + dy^2 <= (ri + rmax[j])^2
                n += Int32(1)
                n <= cap && (idx[n, i] = j)
            end
        end
    end
    count[i] = n
end

"""
    update_neighbors!(nl, cl, bodies, geom, box, range)

Rebuild the neighbour list from the cell list `cl` (which must be current).
"""
function update_neighbors!(nl::NeighborList, cl::CellList, bodies::Bodies, rmax, box::PeriodicBox, range)
    backend = get_backend(bodies.x)
    T = eltype(bodies.x)
    while true
        neighbor_kernel!(backend, WORKGROUP)(nl.idx, nl.count, bodies.x, bodies.y, rmax, cl.perm,
            cl.start, cl.stop, box.Lx, box.Ly, cl.ncx, cl.ncy, cl.wx, cl.wy, T(range),
            Int32(capacity(nl)); ndrange = nbodies(bodies))
        nmax = Int(maximum(nl.count))
        nmax <= capacity(nl) && return nl
        nl.idx = KernelAbstractions.zeros(backend, Int32, 2nmax, nbodies(bodies))
    end
end
