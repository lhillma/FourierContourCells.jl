# Prototype: cells with a nucleus.
#
#   julia -t auto --project=examples examples/nucleus.jl          # live window
#   julia -t auto --project=examples examples/nucleus.jl check    # headless run + diagnostics
#   julia -t auto --project=examples examples/nucleus.jl movie    # writes output/nucleus.mp4
#
# Extension of the Fourier contour cell model of N. Saito and S. Ishihara,
# Sci. Adv. 10, eadi8433 (2024), doi:10.1126/sciadv.adi8433.
#
# Model. Every cell (type 1) contains one nucleus (type 2), a Fourier contour
# body of its own whose `parent` is the cell.
#
#   - Overlap: a nucleus overlapping its own cell costs nothing (`related`
#     coupling 0); overlap with any other cell or nucleus costs J > 0.
#   - Activity: the nucleus has v0 = 0. Its orientation θ diffuses
#     independently of the cell's (rotational noise Dr) but only sets the frame
#     of its shape modes.
#   - Anchoring: a harmonic spring E = (k/2) |r_n - r_c|² between the centres of
#     nucleus and cell, a crude model of the cytoskeleton. The force acts on
#     both centres (equal and opposite, minimum image).
#   - Nucleus stiffness: a stronger line tension on type 2 than on type 1.
#
# The spring is not expressible with the current energy API (BodyTerms depend
# on the shape only, ContactTerms on the penetration depth only), so this
# script adds its gradient between `compute_gradients!` and the integrator in
# a custom time step. See the notes at the end of the file.

using FourierContourCells
using GLMakie
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend
using Random: Xoshiro, randn!

const FCC = FourierContourCells
const CELL, NUCLEUS = 1, 2

# ---------------------------------------------------------------------------
# Initial condition: one round nucleus of radius `ratio·R0` at every cell centre

"""
    with_nuclei(cells; ratio = 0.5, rng) -> (bodies, anchor)

Append one nucleus per cell to the host `Bodies` `cells` (bodies 1:N are the
cells, N+1:2N the nuclei, nucleus N+i belongs to cell i). `anchor[i]` is the
spring partner of body i: the nucleus of a cell, the cell of a nucleus.
"""
function with_nuclei(cells::Bodies; ratio = 0.5, rng = Xoshiro(2))
    N = nbodies(cells)
    coef = hcat(cells.coef, repeat([1.0; zeros(size(cells.coef, 1) - 1)], 1, N))
    bodies = Bodies([cells.x; cells.x], [cells.y; cells.y], [cells.θ; 2π .* rand(rng, N)], coef,
        [cells.R0; ratio .* cells.R0]; type = [fill(CELL, N); fill(NUCLEUS, N)],
        parent = [1:N; 1:N])
    anchor = Int32[(N + 1):(2N); 1:N]
    return bodies, anchor
end

# ---------------------------------------------------------------------------
# Spring between nucleus and cell centres

struct CentreSpring{T}
    k::T
end

# Add the gradient of the centre spring. One work item per body i.
#
# Each anchored pair (i, j = anchor[i]) has energy E = (k/2) |Δ|², Δ = rᵢ - rⱼ
# (minimum image), so ∂E/∂rᵢ = k Δ. The partner j gets -k Δ from its own work
# item (anchor is symmetric), so no atomics are needed. anchor[i] = 0: no spring.
#
#   gx, gy  (in/out) N, ∂E/∂x, ∂E/∂y (accumulated)
#   x, y    N, wrapped body positions
#   anchor  N, spring partner of each body (0: none)
#   k       spring constant;  Lx, Ly: box size
@kernel function centre_spring_kernel!(gx, gy, @Const(x), @Const(y), @Const(anchor), k, Lx, Ly)
    i = @index(Global)
    j = anchor[i]
    if j > 0
        gx[i] += k * FCC.minimum_image(x[i] - x[j], Lx)
        gy[i] += k * FCC.minimum_image(y[i] - y[j], Ly)
    end
end

function add_spring!(sim, s::CentreSpring, anchor)
    b = sim.bodies
    T = eltype(b.x)
    centre_spring_kernel!(get_backend(b.x), 256)(sim.grad.gx, sim.grad.gy, b.x, b.y, anchor,
        T(s.k), sim.box.Lx, sim.box.Ly; ndrange = nbodies(b))
    return sim
end

# Time step: a copy of `FourierContourCells.step!` with the spring inserted
# after the energy gradients. Uses internal (non-public) functions.
function nucleus_step!(sim, s::CentreSpring, anchor, n::Integer = 1)
    for _ in 1:n
        FCC.update_neighbors!(sim)
        compute_gradients!(sim)
        add_spring!(sim, s, anchor)
        randn!(sim.rng, sim.noise)
        FCC.integrate!(sim.bodies, sim.grad, sim.noise, sim.types, sim.dt, sim.box, sim.basis)
        FCC.update_geometry!(sim.geom, sim.basis, sim.bodies)
        sim.t += sim.dt
        sim.step += 1
    end
    return sim
end

# ---------------------------------------------------------------------------
# Diagnostics (host side)

"""
Per nucleus: centre offset |r_n - r_c| from its cell, and protrusion
max_k (|u_k - r_c| - R_c(ψ_k)) of its contour nodes u_k beyond the cell
contour (> 0: the nucleus pokes out of its cell).
"""
function nucleus_diagnostics(sim)
    b = FCC.to_host(sim.bodies)
    R = Array(sim.geom.R)
    lab = Array(sim.geom.lab)
    K = nmodes(sim.basis)
    Lx, Ly = sim.box.Lx, sim.box.Ly
    nuclei = findall(==(NUCLEUS), b.type)
    offset = zeros(length(nuclei))
    protrusion = zeros(length(nuclei))
    for (q, n) in enumerate(nuclei)
        c = b.parent[n]
        offset[q] = hypot(FCC.minimum_image(b.x[n] - b.x[c], Lx), FCC.minimum_image(b.y[n] - b.y[c], Ly))
        p = -Inf
        for k in 1:(sim.basis.nθ)
            φ = FCC.node_angle(sim.basis, k)
            dx = FCC.minimum_image(b.x[n] + R[k, n] * cos(φ) - b.x[c], Lx)
            dy = FCC.minimum_image(b.y[n] + R[k, n] * sin(φ) - b.y[c], Ly)
            p = max(p, hypot(dx, dy) - FCC.contour_radius(lab, c, b.R0[c], K, atan(dy, dx)))
        end
        protrusion[q] = p
    end
    return (; offset, protrusion)
end

mean(v) = sum(v) / length(v)

function report(sim)
    d = nucleus_diagnostics(sim)
    p = Array(shape_index(sim))
    typ = Array(sim.bodies.type)
    println("t = ", round(sim.t; digits = 1),
        "  offset mean/max = ", round(mean(d.offset); digits = 3), "/", round(maximum(d.offset); digits = 3),
        "  protruding nuclei = ", count(>(0), d.protrusion), " (max ", round(maximum(d.protrusion); digits = 3), ")",
        "  ⟨p⟩ cells/nuclei = ", round(mean(p[typ .== CELL]); digits = 3), "/",
        round(mean(p[typ .== NUCLEUS]); digits = 3))
end

# ---------------------------------------------------------------------------
# Plotting: cells coloured by shape index, nuclei drawn on top in one colour
# (their value lies above the colour range and is shown with `highclip`).

const PRANGE = (3.545f0, 4.2f0)
cell_colors(sim) = ifelse.(Array(sim.bodies.type) .== CELL, min.(Array(shape_index(sim)), PRANGE[2]), 10.0)

function nucleus_plot(sim)
    return cellplot(sim; color = cell_colors, colorrange = PRANGE,
        colormap = Makie.cgrad([:gray85, :firebrick]), highclip = :steelblue)
end

# `live_view` and `record_movie` call `step!`, which knows nothing about the
# spring, so the loops are rewritten here.
function nucleus_live(sim, s, anchor; steps_per_frame = 50)
    fig, view = nucleus_plot(sim)
    screen = display(fig)
    while isopen(screen)
        nucleus_step!(sim, s, anchor, steps_per_frame)
        update_view!(view)
        sleep(1 / 120)
    end
end

function nucleus_movie(sim, s, anchor, path; frames = 600, steps_per_frame = 100)
    fig, view = nucleus_plot(sim)
    record(fig, path, 1:frames; framerate = 30) do _
        nucleus_step!(sim, s, anchor, steps_per_frame)
        update_view!(view)
    end
    return path
end

# ---------------------------------------------------------------------------
# Setup: "soft fluid" cell parameters of Saito & Ishihara (2024), Fig. 2C

cells, box = hexagonal_monolayer(32, 32; packing = 0.95)
bodies, anchor_host = with_nuclei(cells; ratio = 0.5)

# overlap couplings indexed (type_i, type_j); same-parent pairs use `related`
J = [1.0 1.0; 1.0 1.0]
Jrelated = [0.0 0.0; 0.0 0.0]

sim = Simulation(bodies, box;
    types = (BodyType(v0 = 0.06, Dr = 0.01, μab = 0.1),    # cell
             BodyType(v0 = 0.0, Dr = 0.01, μab = 0.1)),    # nucleus
    energies = (OnTypes(Tension(0.01), CELL), OnTypes(Tension(0.05), NUCLEUS),
                Overlap(J; related = Jrelated)),
    dt = 0.005)
spring = CentreSpring(1.0)
anchor = FCC.to_backend(get_backend(sim.bodies.x), anchor_host)

mode = get(ARGS, 1, "live")
if mode == "check"
    for _ in 1:10
        nucleus_step!(sim, spring, anchor, 2_000)
        report(sim)
    end
    mkpath(joinpath(@__DIR__, "..", "output"))
    path = joinpath(@__DIR__, "..", "output", "nucleus.png")
    save(path, first(nucleus_plot(sim)))
    println("wrote ", path)
elseif mode == "movie"
    nucleus_step!(sim, spring, anchor, 20_000)
    mkpath(joinpath(@__DIR__, "..", "output"))
    println("wrote ", nucleus_movie(sim, spring, anchor, joinpath(@__DIR__, "..", "output", "nucleus.mp4")))
elseif mode == "live"
    nucleus_live(sim, spring, anchor)
end

# ---------------------------------------------------------------------------
# Notes from this prototype (extensibility test of the library)
#
# Library friction, roughly by importance:
#
# 1. No way to add centre-based (bonded) interactions. BodyTerms see only the
#    shape, ContactTerms only the penetration depth. The spring therefore
#    needs a hand-copied `step!` (`nucleus_step!`) built from non-public
#    functions (`update_neighbors!`, `integrate!`, `update_geometry!`), which
#    silently diverges if `step!` changes. Candidates: a third term kind
#    (e.g. `BondTerm` with a bond list, E(rᵢ - rⱼ)), or a force hook in `step!`.
# 2. `live_view` / `record_movie` hard-code `step!`, so both loops are
#    rewritten here. This would be solved by 1, or by a `step` keyword.
# 3. No parent → child lookup: `Bodies.parent` maps nucleus → cell only; the
#    symmetric `anchor` table is built by hand.
# 4. No helper to combine body sets (`vcat` of `Bodies`) or to build cells
#    with nuclei; `with_nuclei` copies fields by hand.
# 5. Plotting cannot style bodies by type: cells and nuclei share one colour
#    map, so nuclei are drawn via an out-of-range value plus `highclip`, and
#    the cell values must be clamped. Candidates: per-type colours or styles,
#    or a body subset in `cellview!`.
# 6. Writing a kernel needs KernelAbstractions as a direct dependency of the
#    script and internals such as `minimum_image`, `node_angle`,
#    `contour_radius`, `to_host`, `to_backend` and `WORKGROUP`; none is
#    public.
# 7. Observables mix types (`shape_index` covers cells and nuclei together);
#    `nucleus_diagnostics` (offset, protrusion) is a candidate observable.
# 8. Cost: 7.7 ms/step for 1024 cells + 1024 nuclei vs 3.0 ms/step for the
#    cells alone (16 threads). Nuclei enter the neighbour lists of nearby
#    cells, and the contact kernel still evaluates every cell–own-nucleus
#    pair although its coupling is 0.
#
# Modelling observations (to discuss):
#
# - With `related` = 0 the cell membrane does not feel its own nucleus. An
#   elongating cell can therefore narrow past it: after t = 100, one nucleus
#   pokes out of its cell (by 0.007 R₀). Confinement would need an energy for
#   the area of a nucleus *outside* its own cell (d < 0 for the related pair),
#   i.e. an interaction range of about a nucleus diameter.
# - The nucleus adds friction: an isolated cell with nucleus (both μr = 1)
#   moves at v0/2 with a lag |r_n - r_c| = v0 / (2 μr k) = 0.03 R₀. In the
#   dense monolayer the mean offset is ≈ 0.01 R₀.
# - The spring acts on the polar origins r, which differ from the area
#   centroids once modes n ≥ 2 are excited (products of modes create n = 1
#   harmonics in ∮ R³ e_φ dφ).
# - With line tension 0.05 the nuclei stay circular (⟨p⟩ = 3.545); they are
#   rarely touched by other bodies.
