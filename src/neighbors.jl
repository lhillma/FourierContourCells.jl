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
