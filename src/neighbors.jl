"""
    CellList(box, cutoff, N; backend)

Uniform grid for neighbour search. Grid cells are at least `cutoff` wide, so
interacting pairs live in the same or adjacent grid cells. A dimension with
fewer than three grid cells collapses to a single cell, in which case all
bodies along it are checked with the minimum image convention.

Fields:
- `ncx`, `ncy`: number of grid cells along x and y; `wx`, `wy`: their widths
- `cellid`: N, linear grid cell index of each body, `cx + ncx·cy + 1`
- `perm`: N, body indices sorted by `cellid`
- `start`, `stop`: `ncx·ncy`, bodies of grid cell `c` are
  `perm[start[c]:stop[c]]` (empty cells have `start = 1`, `stop = 0`)
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

"""
Grid cell coordinates (0-based) of the wrapped position (`x`, `y`) on a grid
with `cl_ncx × cl_ncy` cells of size `wx × wy`. Positions exactly at the upper
box edge are clamped into the last cell.
"""
@inline grid_coords(cl_ncx, cl_ncy, wx, wy, x, y) =
    (min(unsafe_trunc(Int32, x / wx), cl_ncx - Int32(1)),
     min(unsafe_trunc(Int32, y / wy), cl_ncy - Int32(1)))

# Assign each body to its grid cell. One work item per body i.
#
#   cellid  (out) N, linear grid cell index (1-based) of each body
#   x, y    N, wrapped body positions
#   ncx, ncy, wx, wy  grid size and cell widths (see CellList)
@kernel function cell_index_kernel!(cellid, @Const(x), @Const(y), ncx, ncy, wx, wy)
    @inbounds begin
        i = @index(Global)
        cx, cy = grid_coords(ncx, ncy, wx, wy, x[i], y[i])
        cellid[i] = cx + ncx * cy + Int32(1)
    end
end

# Find the range of each grid cell in the sorted body list. One work item per
# position s in the sorted order: s starts a cell if the previous body lies in
# a different cell and ends it if the next one does (bodies of one cell are
# contiguous after sorting by cellid).
#
#   start, stop  (out) ncx·ncy, first/last sorted position of each grid cell
#                (must be reset to 1/0 beforehand so empty cells are empty ranges)
#   cellid       N, grid cell of each body
#   perm         N, body indices sorted by cellid
#   N            number of bodies
@kernel function cell_bounds_kernel!(start, stop, @Const(cellid), @Const(perm), N)
    @inbounds begin
        s = @index(Global)
        c = cellid[perm[s]]
        if s == 1 || cellid[perm[s - 1]] != c
            start[c] = s
        end
        if s == N || cellid[perm[s + 1]] != c
            stop[c] = s
        end
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
    NeighborList(N, T; skin = 0, capacity = 32, backend)

Per-body list of potentially interacting bodies, stored as a padded
`capacity × N` matrix `idx` with `count[i]` valid entries in column `i`.
Body j is listed for body i if their centres are closer than
rmax_i + rmax_j + range + skin, which no contour point of i can reach otherwise.
The capacity grows automatically when exceeded.

Verlet skin: the list stays valid while no pair can have closed the `skin`
gap. With the drift δᵢ = |rᵢ - rᵢ⁰| + max(0, rmaxᵢ - rmaxᵢ⁰) since the last
build (positions `x0`, `y0`, bounds `rmax0`), an unlisted pair is still out of
range while δᵢ + δⱼ ≤ skin, so the list is rebuilt once some 2δᵢ > skin (each
body checks itself and raises a flag, see `skin_check!`). With `skin = 0` it
is rebuilt every step. `builds` counts the builds.
"""
mutable struct NeighborList{T, MI <: AbstractMatrix{Int32}, VI <: AbstractVector{Int32}, V <: AbstractVector{T}}
    idx::MI
    count::VI
    const skin::T
    const x0::V
    const y0::V
    const rmax0::V
    valid::Bool
    builds::Int
end

function NeighborList(N::Integer, ::Type{T} = Float64; skin = 0, capacity::Integer = 32,
        backend = CPU()) where {T}
    skin >= 0 || throw(ArgumentError("skin must be non-negative"))
    vec() = KernelAbstractions.zeros(backend, T, N)
    return NeighborList(KernelAbstractions.zeros(backend, Int32, capacity, N),
        KernelAbstractions.zeros(backend, Int32, N), T(skin), vec(), vec(), vec(), false, 0)
end

capacity(nl::NeighborList) = size(nl.idx, 1)

# Build the per-body neighbour list. One work item per body i: loop over the
# 3×3 grid cells around i (fewer for collapsed dimensions) and keep every j ≠ i
# with centre distance ≤ rmaxᵢ + rmaxⱼ + range. Otherwise no contour point of i
# (at distance ≤ rmaxᵢ from rᵢ) can come within rmaxⱼ + range of rⱼ, i.e. within
# range of body j. Callers include the Verlet skin in range. Entries beyond the
# capacity are counted but not stored, so the caller can detect overflow and
# retry with a larger matrix.
#
#   idx      (out) cap × N, neighbour indices of body i in column i
#   count    (out) N, number of neighbours found (may exceed cap)
#   x, y     N, wrapped body positions
#   rmax     N, upper bound of each body's contour radius
#   perm, start, stop  cell list (see CellList)
#   Lx, Ly   box size (minimum image convention)
#   ncx, ncy, wx, wy   grid size and cell widths
#   range    interaction range of the contact terms beyond contact, plus skin
#   cap      capacity, size(idx, 1)
@kernel function neighbor_kernel!(idx, count, @Const(x), @Const(y), @Const(rmax), @Const(perm),
        @Const(start), @Const(stop), Lx, Ly, ncx, ncy, wx, wy, range, cap)
    @inbounds begin
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
end

# Verlet skin check (see NeighborList). One work item per body i: compute the
# drift since the last neighbour list build,
#   δᵢ = |rᵢ - rᵢ⁰| + max(0, rmaxᵢ - rmaxᵢ⁰),
# and raise the rebuild flag if 2δᵢ > skin. All writers store the same value,
# so no reduction or atomics are needed.
#
#   status      (in/out) run status, entry STATUS_REBUILD set to 1 if needed
#   x, y, rmax  N, current wrapped positions and contour radius bounds
#   x0, y0, rmax0  N, the same at the last build
#   Lx, Ly      box size (minimum image of the displacement); skin: Verlet skin
@kernel function skin_check_kernel!(status, @Const(x), @Const(y), @Const(rmax), @Const(x0),
        @Const(y0), @Const(rmax0), Lx, Ly, skin)
    @inbounds begin
        i = @index(Global)
        dx = minimum_image(x[i] - x0[i], Lx)
        dy = minimum_image(y[i] - y0[i], Ly)
        δ = sqrt(dx^2 + dy^2) + max(zero(dx), rmax[i] - rmax0[i])
        2δ > skin && (status[STATUS_REBUILD] = Int32(1))
    end
end

"Raise the rebuild flag in `status` if the Verlet skin of `nl` is exceeded (no host sync)."
function skin_check!(status, nl::NeighborList, bodies::Bodies, rmax, box::PeriodicBox)
    skin_check_kernel!(get_backend(status), WORKGROUP)(status, bodies.x, bodies.y, rmax, nl.x0,
        nl.y0, nl.rmax0, box.Lx, box.Ly, nl.skin; ndrange = nbodies(bodies))
    return status
end

"""
    update_neighbors!(nl, cl, bodies, rmax, box, range)

Rebuild the neighbour list from the cell list `cl` (which must be current)
and record the reference state of the Verlet skin.
"""
function update_neighbors!(nl::NeighborList, cl::CellList, bodies::Bodies, rmax, box::PeriodicBox, range)
    backend = get_backend(bodies.x)
    T = eltype(bodies.x)
    while true
        neighbor_kernel!(backend, WORKGROUP)(nl.idx, nl.count, bodies.x, bodies.y, rmax, cl.perm,
            cl.start, cl.stop, box.Lx, box.Ly, cl.ncx, cl.ncy, cl.wx, cl.wy, T(range) + nl.skin,
            Int32(capacity(nl)); ndrange = nbodies(bodies))
        nmax = Int(maximum(nl.count))
        nmax <= capacity(nl) && break
        nl.idx = KernelAbstractions.zeros(backend, Int32, 2nmax, nbodies(bodies))
    end
    if nl.skin > 0
        copyto!(nl.x0, bodies.x)
        copyto!(nl.y0, bodies.y)
        copyto!(nl.rmax0, rmax)
    end
    nl.valid = true
    nl.builds += 1
    return nl
end
