"""
    Simulation(bodies, box; types, energies, bonds = (), dt = 0.005, nθ = 40,
               backend = CPU(), T = Float64, seed = 1, rng = nothing)

Complete simulation state. `bodies` (host or device [`Bodies`](@ref)) are
converted to element type `T` and moved to the KernelAbstractions `backend`.

- `types`: tuple of [`BodyType`](@ref), indexed by `bodies.type`
- `energies`: tuple of [`BodyTerm`](@ref)s, [`ContactTerm`](@ref)s and
  [`BondTerm`](@ref)s
- `bonds`: pairs `(i, j)` of bonded bodies, on which the bond terms act
  (e.g. [`parent_bonds`](@ref)`(bodies)`)
- `nθ`: number of contour quadrature nodes
- `rng`: random number generator compatible with `backend`
  (default: `backend_rng(backend, seed)`)

Body types with `drag = :parent` (see [`BodyType`](@ref)) require every body
of that type to have a parent with substrate drag.
"""
mutable struct Simulation{T, Ba <: FourierBasis{T}, Bo <: Bodies, G <: Geometry,
        Gr <: Gradients, CL <: CellList, NL <: NeighborList, ChL <: ChildList, BL <: BondList, BT <: Tuple,
        CT <: Tuple, BoT <: Tuple, TT <: Tuple, R, Mat}
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
    const rng::R
    const noise::Mat
    const disp::Mat
    dt::T
    t::T
    step::Int
end

function Simulation(bodies::Bodies, box::PeriodicBox; types, energies = (), bonds = (), dt = 0.005,
        nθ::Integer = 40, backend = CPU(), T::Type = Float64, seed = 1, rng = nothing)
    types = map(BodyType{T}, Tuple(types))
    energies = Tuple(energies)
    body_terms = filter(is_body_term, energies)
    contact_terms = filter(is_contact_term, energies)
    bond_terms = filter(is_bond_term, energies)
    length(body_terms) + length(contact_terms) + length(bond_terms) == length(energies) ||
        throw(ArgumentError("energy terms must be BodyTerms, ContactTerms or BondTerms"))

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
    cutoff = 2 * maximum(host.R0) * (1 + SHAPE_BOUND) + contact_range(contact_terms)
    cells = CellList(box, cutoff, nbodies(dev); backend)
    rng = rng === nothing ? backend_rng(backend, seed) : rng
    noise = KernelAbstractions.zeros(backend, T, 3, nbodies(dev))
    disp = KernelAbstractions.zeros(backend, T, 2, nbodies(dev))

    neighbors = NeighborList(nbodies(dev); backend)
    children = ChildList(host.parent; backend)
    bondlist = BondList(bonds, nbodies(host); backend)
    sim = Simulation(basis, box, dev, Geometry(basis, dev), Gradients(basis, dev), cells, neighbors,
        children, bondlist, body_terms, contact_terms, bond_terms, types, rng, noise, disp, T(dt),
        zero(T), 0)
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

"Default random number generator for a backend."
backend_rng(::CPU, seed) = Random.Xoshiro(seed)

nbodies(sim::Simulation) = nbodies(sim.bodies)

"""
    update_neighbors!(sim)

Rebuild the cell list and the per-body neighbour list for the current state.
"""
function update_neighbors!(sim::Simulation)
    update_celllist!(sim.cells, sim.bodies)
    update_neighbors!(sim.neighbors, sim.cells, sim.bodies, sim.geom.rmax, sim.box,
        contact_range(sim.contact_terms))
    return sim
end

"""
    compute_gradients!(sim)

Evaluate ∂E/∂(r, θ, aₙ, bₙ) of all energy terms for the current state.
Requires up-to-date geometry and neighbour lists (see [`update_neighbors!`](@ref)).
"""
function compute_gradients!(sim::Simulation)
    (; basis, box, bodies, geom, grad, neighbors) = sim
    reset!(grad)
    body_terms!(grad, sim.body_terms, geom, bodies, basis)
    if !isempty(sim.contact_terms)
        contact_pressure!(grad.P, sim.contact_terms, geom, bodies, neighbors, box, basis)
        contact_gradient!(grad, geom, basis)
    end
    bond_gradient!(grad, sim.bond_terms, bodies, sim.bonds, box)
    pullback!(grad, geom, bodies, basis)
    return grad
end

"""
    step!(sim, n = 1)

Advance the simulation by `n` Euler–Maruyama steps.
"""
function step!(sim::Simulation, n::Integer = 1)
    for _ in 1:n
        isempty(sim.contact_terms) || update_neighbors!(sim)
        compute_gradients!(sim)
        randn!(sim.rng, sim.noise)
        integrate!(sim.bodies, sim.grad, sim.disp, sim.noise, sim.children, sim.types, sim.dt,
            sim.box, sim.basis)
        update_geometry!(sim.geom, sim.basis, sim.bodies)
        sim.t += sim.dt
        sim.step += 1
    end
    return sim
end

"""
    run!(sim, nsteps; every = 0, callback = nothing)

Advance `nsteps` steps, calling `callback(sim)` every `every` steps.
"""
function run!(sim::Simulation, nsteps::Integer; every::Integer = 0, callback = nothing)
    if callback === nothing || every <= 0
        step!(sim, nsteps)
    else
        done = 0
        while done < nsteps
            n = min(every, nsteps - done)
            step!(sim, n)
            done += n
            callback(sim)
        end
    end
    return sim
end

function Base.show(io::IO, sim::Simulation{T}) where {T}
    print(io, "Simulation{$T}(N=$(nbodies(sim)), $(sim.basis), box=$(sim.box.Lx)×$(sim.box.Ly), ",
        "t=$(sim.t), backend=$(get_backend(sim.bodies.x)))")
end
