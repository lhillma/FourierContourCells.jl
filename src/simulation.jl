"""
    Simulation(bodies, box; types, energies, bonds = (), dt = 0.005, nθ = 40,
               backend = CPU(), T = Float64, seed = 1, skin = 0.3, hook! = nothing,
               layout = default_layout(backend))

Complete simulation state. `bodies` (host or device [`Bodies`](@ref)) are
converted to element type `T` and moved to the KernelAbstractions `backend`.

- `types`: tuple of [`BodyType`](@ref), indexed by `bodies.type`
- `energies`: tuple of [`BodyTerm`](@ref)s, [`ContactTerm`](@ref)s and
  [`BondTerm`](@ref)s
- `T`: floating-point precision of the state; the parameters of `types` and
  `energies` are converted to it (see [`convert_precision`](@ref))
- `bonds`: pairs `(i, j)` of bonded bodies, on which the bond terms act
  (e.g. [`parent_bonds`](@ref)`(bodies)`)
- `nθ`: number of contour quadrature nodes
- `skin`: Verlet skin of the neighbour list (length). The list is rebuilt
  only when a body has drifted (moved, or grown its contour bound) by more
  than skin/2 since the last build; `skin = 0` rebuilds every step. Larger
  skins mean fewer rebuilds but longer neighbour lists.
- `seed`: key of the counter-based Philox random numbers. The noise of body
  `i` in step `n` depends only on (seed, i, n), so it is the same on every
  backend and independent of the number of threads.
- `layout`: [`KernelLayout`](@ref) of the single-body term kernel
  (`BodyParallel()` on the CPU, `NodeParallel()` on GPUs). Both give the same
  results up to round-off.

- `hook!`: optional function `hook!(sim)` called in every [`step!`](@ref)
  after the energy gradients `sim.grad` are computed and before the
  integration step. It may modify the gradients, e.g. to add forces that no
  energy term expresses yet. **Meant for prototyping only**: it relies on
  internal fields, which may change, and is not part of the GPU-ready API.
  Once a prototype works, move it into a proper term.

Body types with `drag = :parent` (see [`BodyType`](@ref)) require every body
of that type to have a parent with substrate drag.
"""
mutable struct Simulation{T, Ba <: FourierBasis{T}, Bo <: Bodies, G <: Geometry,
        Gr <: Gradients, CL <: CellList, NL <: NeighborList, ChL <: ChildList, BL <: BondList, BT <: Tuple,
        CT <: Tuple, BoT <: Tuple, TT <: Tuple, Mat, H, L <: KernelLayout, VS <: AbstractVector{Int32}, VC <: AbstractVector{Int64}}
    const basis::Ba
    const box::PeriodicBox{T}
    const bodies::Bo
    const geom::G
    const grad::Gr
    const cells::CL
    const neighbors::NL
    const children::ChL
    const bonds::BL
    const body_terms::BT
    const contact_terms::CT
    const bond_terms::BoT
    const types::TT
    const key::NTuple{2, UInt32}
    const disp::Mat
    const hook::H
    const layout::L
    const status::VS
    const clock::VC
    const graph::Base.RefValue{Any}
    dt::T
    t::T
    step::Int
end

function Simulation(bodies::Bodies, box::PeriodicBox; types, energies = (), bonds = (), dt = 0.005,
        nθ::Integer = 40, backend = CPU(), T::Type = Float64, seed::Integer = 1, skin::Real = 0.3,
        hook! = nothing, layout::KernelLayout = default_layout(backend))
    types = map(BodyType{T}, Tuple(types))
    energies = Tuple(energies)
    all(t -> is_body_term(t) || is_contact_term(t) || is_bond_term(t), energies) ||
        throw(ArgumentError("energy terms must be BodyTerms, ContactTerms or BondTerms"))
    energies = map(t -> convert_precision(T, t), energies)
    body_terms = filter(is_body_term, energies)
    contact_terms = filter(is_contact_term, energies)
    bond_terms = filter(is_bond_term, energies)

    host = to_host(bodies)
    maximum(host.type) <= length(types) || throw(ArgumentError("body type without BodyType entry"))
    host = Bodies(map(f -> _convert_eltype(T, getfield(host, f)), fieldnames(Bodies))...)
    check_parents(host, types)
    dev = to_backend(backend, host)

    K = size(dev.coef, 1) ÷ 2
    basis = FourierBasis{T}(; maxmode = K + 1, nθ, backend)
    size(dev.coef, 1) == ncoef(basis) || throw(DimensionMismatch("coefficient rows must be odd: [a₀; aₙ; bₙ]"))
    box = PeriodicBox{T}(box)

    # Largest possible extent of a body under the shape bound, plus range of contact terms
    cutoff = 2 * maximum(host.R0) * (1 + SHAPE_BOUND) + contact_range(contact_terms) + skin
    cells = CellList(box, cutoff, nbodies(dev); backend)
    disp = KernelAbstractions.zeros(backend, T, 3, nbodies(dev))

    neighbors = NeighborList(nbodies(dev), T; skin, backend)
    children = ChildList(host.parent; backend)
    bondlist = BondList(bonds, nbodies(host); backend)
    sim = Simulation(basis, box, dev, Geometry(basis, dev), Gradients(basis, dev), cells, neighbors,
        children, bondlist, body_terms, contact_terms, bond_terms, types, philox_key(seed), disp, hook!, layout,
        KernelAbstractions.zeros(backend, Int32, STATUS_LENGTH), KernelAbstractions.zeros(backend, Int64, 1),
        Ref{Any}(nothing),
        T(dt), zero(T), 0)
    enforce_shape!(dev, basis)
    update_geometry!(sim.geom, basis, dev)
    return sim
end

"Check parent ids, and that bodies with parent drag have a parent with substrate drag."
function check_parents(b::Bodies, types)
    N = nbodies(b)
    for i in 1:N
        p = b.parent[i]
        1 <= p <= N || throw(ArgumentError("body $i has parent $p outside 1:$N"))
        types[b.type[i]].parent_drag || continue
        p == i && throw(ArgumentError("body $i has drag = :parent but is its own parent"))
        types[b.type[p]].parent_drag &&
            throw(ArgumentError("body $i has drag = :parent, but so has its parent $p (one level only)"))
    end
end

_convert_eltype(::Type{T}, A::AbstractArray{<:AbstractFloat}) where {T} = T.(A)
_convert_eltype(::Type, A::AbstractArray) = A

nbodies(sim::Simulation) = nbodies(sim.bodies)

"""
    update_neighbors!(sim; force = false)

Rebuild the cell list and the per-body neighbour list if the Verlet skin no
longer covers the current state (see `skin` in [`Simulation`](@ref)), or
always with `force = true`. Requires up-to-date geometry.
"""
function update_neighbors!(sim::Simulation; force::Bool = false)
    nl = sim.neighbors
    if !force && nl.valid && nl.skin > 0
        skin_check!(sim.status, nl, sim.bodies, sim.geom.rmax, sim.box)
        to_host(sim.status)[STATUS_REBUILD] == 0 && return sim
    end
    update_celllist!(sim.cells, sim.bodies)
    update_neighbors!(nl, sim.cells, sim.bodies, sim.geom.rmax, sim.box, contact_range(sim.contact_terms))
    fill!(sim.status, 0)
    return sim
end

"""
    compute_gradients!(sim)

Evaluate ∂E/∂(r, θ, aₙ, bₙ) of all energy terms for the current state.
Requires up-to-date geometry and neighbour lists (see [`update_neighbors!`](@ref)).
"""
function compute_gradients!(sim::Simulation; frame::Bool = true, status = nothing)
    (; basis, box, bodies, geom, grad, neighbors) = sim
    body_terms!(grad, sim.body_terms, geom, bodies, basis, sim.layout; status)  # initialises grad
    isempty(sim.contact_terms) ||
        contact_gradients!(grad, sim.contact_terms, geom, bodies, neighbors, box, basis; status)
    bond_gradient!(grad, sim.bond_terms, bodies, sim.bonds, box; status)
    pullback!(grad, geom, bodies, basis; frame)
    return grad
end

"""
    step!(sim, n = 1)

Advance the simulation by `n` Euler–Maruyama steps.
"""
function step!(sim::Simulation, n::Integer = 1)
    for _ in 1:n
        isempty(sim.contact_terms) || update_neighbors!(sim)
        advance!(sim)
        sim.t += sim.dt
        sim.step += 1
    end
    return sim
end

# One step after the neighbour list update: gradients, hook, integration and
# geometry. Only device operations apart from the hook. Per-body work is
# fused into the integration kernels (see euler_maruyama_kernel!): noise
# generated on the fly (draw index from the device clock), the body-frame
# gradient (unless a hook may modify sim.grad first), the lab-frame geometry
# and the Verlet skin check, which raises the rebuild flag for the next step.
function advance!(sim::Simulation)
    hooked = sim.hook !== nothing
    compute_gradients!(sim; frame = hooked, status = sim.status)
    hooked && sim.hook(sim)
    skin = isempty(sim.contact_terms) ? nothing : skin_args(sim.neighbors)
    integrate!(sim.bodies, sim.grad, sim.disp, PhiloxNoise(sim.key, sim.clock), sim.children,
        sim.types, sim.dt, sim.box, sim.basis; geom = sim.geom, skin, frame = !hooked,
        clock = sim.clock, status = sim.status)
    synthesize!(sim.geom, sim.basis)
    return sim
end

"""
    device_step!(sim)

One step without host synchronisation, for capture in a CUDA graph: the skin
check (fused into the update of the previous step) only raises the rebuild
flag in `sim.status`, which freezes this and all following steps (see
`isactive`) until the host rebuilds the neighbour list.
Requires a valid neighbour list with `skin > 0` and no hook; `sim.t` and
`sim.step` are not updated (the device clock is).
"""
device_step!(sim::Simulation) = advance!(sim)

"Whether `sim` can run in graph mode (see [`run!`](@ref))."
graph_capable(sim::Simulation) =
    sim.hook === nothing && (isempty(sim.contact_terms) || sim.neighbors.skin > 0)

"""
    run!(sim, nsteps; every = 0, callback = nothing, graph = true)

Advance `nsteps` steps, calling `callback(sim)` every `every` steps.

With `graph = true`, backends that support it run the steps between callbacks
as captured graphs (CUDA, via the CUDA extension: one launch per step and one
host synchronisation per chunk of steps instead of one per step). Results are
identical to [`step!`](@ref). Graph mode needs `hook! = nothing` and, with
contact terms, a Verlet `skin > 0`; otherwise `run!` steps eagerly.
"""
function run!(sim::Simulation, nsteps::Integer; every::Integer = 0, callback = nothing,
        graph::Bool = true)
    advance(n) = graph && graph_capable(sim) ? run_steps!(get_backend(sim.bodies.x), sim, n) : step!(sim, n)
    if callback === nothing || every <= 0
        advance(nsteps)
    else
        done = 0
        while done < nsteps
            n = min(every, nsteps - done)
            advance(n)
            done += n
            callback(sim)
        end
    end
    return sim
end

"Run `n` steps on `backend`; extensions add graph-based methods (default: [`step!`](@ref))."
run_steps!(backend, sim::Simulation, n::Integer) = step!(sim, n)

function Base.show(io::IO, sim::Simulation{T}) where {T}
    print(io, "Simulation{$T}(N=$(nbodies(sim)), $(sim.basis), box=$(sim.box.Lx)×$(sim.box.Ly), ",
        "t=$(sim.t), backend=$(get_backend(sim.bodies.x)))")
end
