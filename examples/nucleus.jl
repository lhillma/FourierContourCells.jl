# Prototype: cells with a nucleus.
#
#   julia -t auto --project=examples examples/nucleus.jl          # live window
#   julia -t auto --project=examples examples/nucleus.jl check    # headless run + diagnostics
#                                        (append `substrate` for substrate drag of nuclei)
#   julia -t auto --project=examples examples/nucleus.jl movie    # writes output/nucleus.mp4
#   julia -t auto --project=examples examples/nucleus.jl test     # isolated-pair checks
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
#   - Drag: the cell has drag ζ_c = 1/μ_c against the substrate; the nucleus
#     has drag ζ_i = 1/μ_i against the cytoplasm, i.e. relative to its cell,
#     with the reaction on the cell. The force balances
#         ζ_c ṙ_c - ζ_i (ṙ_n - ṙ_c) = F_c,    ζ_i (ṙ_n - ṙ_c) = F_n
#     give the explicit velocities
#         ṙ_c = μ_c (F_c + F_n),    ṙ_n = ṙ_c + μ_i F_n.
#     The cell carries its nucleus's forces with its own drag, so the spring
#     is internal (its forces cancel in F_c + F_n) and only relaxes the
#     nucleus towards the cell centre at rate μ_i k. An isolated cell moves at
#     v0 with zero lag. Without relative drag (`drag = nothing`) both bodies
#     have substrate drag, which slows an isolated cell to
#     v0 μ_n / (μ_c + μ_n).
#   - Nucleus stiffness: a stronger line tension on type 2 than on type 1.
#
# Neither the spring nor the relative drag fits the current energy API
# (BodyTerms depend on the shape only, ContactTerms on the penetration depth
# only), so this script modifies the centre gradients between
# `compute_gradients!` and the integrator in a custom time step. See the notes
# at the end of the file.

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

"""
    RelativeDrag(μi)

Drag of a nucleus relative to its cell, with mobility `μi` (see the header).
"""
struct RelativeDrag{T}
    μi::T
end

# Rewrite the centre gradients so that the integrator, which moves body i by
# dt (v0 e(θᵢ) - μr gᵢ), produces the relative-drag velocities. One work item
# per body i; g = ∂E/∂r = -F (spring included).
#
#   cell c with nucleus n:  g_c ← g_c + g_n, so that
#       v_c = v0_c e(θ_c) - μ_c (g_c + g_n)
#   nucleus n of cell c:    target v_n = v_c - μ_i g_n; the integrator gives
#       v0_n e(θ_n) - μr_n g, so  g ← (v0_n e(θ_n) - v_n) / μr_n
#   other bodies:           unchanged
#
# The work items read gradients of their partner, so the results go to
# separate buffers (hx, hy) and are copied back afterwards. θ is read before
# the integrator updates it. Translational noise (Dt) of the cell is not
# passed on to the nucleus.
#
#   hx, hy      (out) N, rewritten ∂E/∂x, ∂E/∂y
#   gx, gy      N, ∂E/∂x, ∂E/∂y from the energy terms and the spring
#   θ, type     N, orientations and body types
#   anchor      N, partner of each body (cell ↔ nucleus; 0: none)
#   types       tuple of BodyType;  nucleus: type index of nuclei
#   μi          mobility of a nucleus relative to its cell
@kernel function relative_drag_kernel!(hx, hy, @Const(gx), @Const(gy), @Const(θ), @Const(type),
        @Const(anchor), types, nucleus, μi)
    i = @index(Global)
    j = anchor[i]
    if j == 0
        hx[i] = gx[i]
        hy[i] = gy[i]
    elseif type[i] != nucleus
        hx[i] = gx[i] + gx[j]
        hy[i] = gy[i] + gy[j]
    else
        pc = types[type[j]]
        pn = types[type[i]]
        sc, cc = sincos(θ[j])
        vx = pc.v0 * cc - pc.μr * (gx[j] + gx[i]) - μi * gx[i]
        vy = pc.v0 * sc - pc.μr * (gy[j] + gy[i]) - μi * gy[i]
        sn, cn = sincos(θ[i])
        hx[i] = (pn.v0 * cn - vx) / pn.μr
        hy[i] = (pn.v0 * sn - vy) / pn.μr
    end
end

"Work buffers for `relative_drag!`."
drag_buffers(sim) = (similar(sim.bodies.x), similar(sim.bodies.x))

function relative_drag!(sim, d::RelativeDrag, anchor, (hx, hy))
    b, g = sim.bodies, sim.grad
    T = eltype(b.x)
    relative_drag_kernel!(get_backend(b.x), 256)(hx, hy, g.gx, g.gy, b.θ, b.type, anchor,
        sim.types, Int32(NUCLEUS), T(d.μi); ndrange = nbodies(b))
    copyto!(g.gx, hx)
    copyto!(g.gy, hy)
    return sim
end

"""
    NucleusModel(spring, drag, anchor, buffers)

Everything the custom time step adds to `sim`: the centre spring, the
relative drag (`nothing`: substrate drag for nuclei too), the partner table
(on the backend) and work buffers. Build with [`nucleus_model`](@ref).
"""
struct NucleusModel{S, D, A, B}
    spring::S
    drag::D
    anchor::A
    buffers::B
end

"Set up a `NucleusModel` for `sim` from the host partner table of `with_nuclei`."
function nucleus_model(sim, spring, drag, anchor_host)
    anchor = FCC.to_backend(get_backend(sim.bodies.x), anchor_host)
    return NucleusModel(spring, drag, anchor, drag_buffers(sim))
end

# Time step: a copy of `FourierContourCells.step!` with the spring and the
# relative drag inserted after the energy gradients. Uses internal
# (non-public) functions.
function nucleus_step!(sim, m::NucleusModel, n::Integer = 1)
    for _ in 1:n
        isempty(sim.contact_terms) || FCC.update_neighbors!(sim)
        compute_gradients!(sim)
        add_spring!(sim, m.spring, m.anchor)
        m.drag === nothing || relative_drag!(sim, m.drag, m.anchor, m.buffers)
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
# spring and the drag, so the loops are rewritten here.
function nucleus_live(sim, m; steps_per_frame = 50)
    fig, view = nucleus_plot(sim)
    screen = display(fig)
    while isopen(screen)
        nucleus_step!(sim, m, steps_per_frame)
        update_view!(view)
        sleep(1 / 120)
    end
end

function nucleus_movie(sim, m, path; frames = 600, steps_per_frame = 100)
    fig, view = nucleus_plot(sim)
    record(fig, path, 1:frames; framerate = 30) do _
        nucleus_step!(sim, m, steps_per_frame)
        update_view!(view)
    end
    return path
end

# ---------------------------------------------------------------------------
# Parameters: "soft fluid" cells of Saito & Ishihara (2024), Fig. 2C

# overlap couplings indexed (type_i, type_j); same-parent pairs use `related`
const J = [1.0 1.0; 1.0 1.0]
const JRELATED = [0.0 0.0; 0.0 0.0]
const ENERGIES = (OnTypes(Tension(0.01), CELL), OnTypes(Tension(0.05), NUCLEUS),
    Overlap(J; related = JRELATED))
cell_type(; v0 = 0.06, Dr = 0.01) = BodyType(; v0, Dr, μab = 0.1)
nucleus_type(; Dr = 0.01) = BodyType(; v0 = 0.0, Dr, μab = 0.1)
const SPRING = CentreSpring(1.0)
const DRAG = RelativeDrag(1.0)

# ---------------------------------------------------------------------------
# Checks on an isolated cell–nucleus pair (no noise, no contacts)

"Cell at the box centre, nucleus displaced by `δ0` along x; orientation θ = 0."
function isolated_pair(; v0, drag, δ0 = 0.0, L = 20.0)
    cells = Bodies([L / 2], [L / 2], [0.0], [1.0; zeros(10)][:, :], [1.0])
    bodies, anchor = with_nuclei(cells)
    bodies.x[2] += δ0
    bodies.θ[2] = 0.0
    sim = Simulation(bodies, PeriodicBox(L, L); types = (cell_type(; v0, Dr = 0.0),
        nucleus_type(; Dr = 0.0)), energies = ENERGIES, dt = 0.005)
    return sim, nucleus_model(sim, SPRING, drag, anchor)
end

"Run `t` time units; return cell speed (last half) and final lag x_c - x_n."
function drift(sim, m, t)
    nhalf = round(Int, t / 2 / sim.dt)
    nucleus_step!(sim, m, nhalf)
    x0 = unwrapped_positions(sim)[1, 1]
    nucleus_step!(sim, m, nhalf)
    X = unwrapped_positions(sim)
    return (X[1, 1] - x0) / (nhalf * sim.dt), X[1, 1] - X[1, 2]
end

function check(name, value, expected; atol)
    ok = isapprox(value, expected; atol)
    println(ok ? "  pass  " : "  FAIL  ", rpad(name, 44), "got ", value, ", expected ", expected)
    return ok
end

function run_tests()
    v0, k, μi = 0.06, SPRING.k, DRAG.μi
    ok = true
    println("isolated pair, relative drag: the composite moves at v0, no lag")
    sim, m = isolated_pair(; v0, drag = DRAG)
    V, δ = drift(sim, m, 50.0)
    ok &= check("speed", V, v0; atol = 1e-12)
    ok &= check("lag", δ, 0.0; atol = 1e-12)

    println("isolated pair, substrate drag: V = v0/2, lag v0/(2k)")
    sim, m = isolated_pair(; v0, drag = nothing)
    V, δ = drift(sim, m, 50.0)
    ok &= check("speed", V, v0 / 2; atol = 1e-9)
    ok &= check("lag", δ, v0 / (2k); atol = 1e-9)

    println("relaxation, relative drag: the spring is internal, the cell stays put")
    δ0, n = 0.1, 400
    sim, m = isolated_pair(; v0 = 0.0, drag = DRAG, δ0)
    xc = sim.bodies.x[1]
    nucleus_step!(sim, m, n)
    ok &= check("cell displacement", sim.bodies.x[1] - xc, 0.0; atol = 1e-14)
    # explicit Euler of δ̇ = -μi k δ
    ok &= check("nucleus offset", sim.bodies.x[2] - sim.bodies.x[1], δ0 * (1 - μi * k * sim.dt)^n;
        atol = 1e-14)
    ok || error("nucleus checks failed")
    return nothing
end

# ---------------------------------------------------------------------------
# Monolayer

function monolayer(; drag = DRAG)
    cells, box = hexagonal_monolayer(32, 32; packing = 0.95)
    bodies, anchor = with_nuclei(cells; ratio = 0.5)
    sim = Simulation(bodies, box; types = (cell_type(), nucleus_type()), energies = ENERGIES,
        dt = 0.005)
    return sim, nucleus_model(sim, SPRING, drag, anchor)
end

"Mean cell displacement |Δr| over the whole run, for comparing drag models."
mean_cell_displacement(sim, X0) =
    mean(vec(sqrt.(sum(abs2, (unwrapped_positions(sim) .- X0)[:, Array(sim.bodies.type) .== CELL]; dims = 1))))

mode = get(ARGS, 1, "live")
if mode == "test"
    run_tests()
elseif mode == "check"
    drag = get(ARGS, 2, "relative") == "substrate" ? nothing : DRAG
    sim, m = monolayer(; drag)
    X0 = unwrapped_positions(sim)
    for _ in 1:10
        nucleus_step!(sim, m, 2_000)
        report(sim)
    end
    println("mean cell displacement: ", round(mean_cell_displacement(sim, X0); digits = 3))
    mkpath(joinpath(@__DIR__, "..", "output"))
    path = joinpath(@__DIR__, "..", "output", "nucleus.png")
    save(path, first(nucleus_plot(sim)))
    println("wrote ", path)
elseif mode == "movie"
    sim, m = monolayer()
    nucleus_step!(sim, m, 20_000)
    mkpath(joinpath(@__DIR__, "..", "output"))
    println("wrote ", nucleus_movie(sim, m, joinpath(@__DIR__, "..", "output", "nucleus.mp4")))
elseif mode == "live"
    sim, m = monolayer()
    nucleus_live(sim, m)
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
#    The relative drag is not an energy at all but a change of the mobility
#    (velocities from forces), so it needs the same hook, or a per-type
#    option "drag relative to parent" in the integrator. Emulating it by
#    rewriting gradients works (the integrator is r += dt (v0 e - μr g)) but
#    needs scratch buffers and couples the script to the integrator's form.
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
#   pokes out of its cell. With relative drag the cells deform more (see
#   below) and 7 nuclei protrude at t = 100 (by up to 0.027 R₀). Confinement
#   would need an energy for the area of a nucleus *outside* its own cell
#   (d < 0 for the related pair), i.e. an interaction range of about a
#   nucleus diameter.
# - Drag (t = 100, same seed; mean cell displacement, ⟨p⟩ of cells):
#     cells only, no nuclei            0.926   3.821
#     nuclei with relative drag        0.930   3.825   mean offset < 0.001 R₀
#     nuclei with substrate drag       0.594   3.766   mean offset ≈ 0.013 R₀
#   Substrate drag on the nucleus double-counts the friction of the cell's
#   footprint: an isolated pair (both μr = 1) moves at v0/2 with a lag of
#   v0 / (2 μr k) = 0.03 R₀, independent of k. With relative drag the
#   composite moves at v0 without lag, and the monolayer matches the
#   cells-only model, since the nuclei are rarely touched by other bodies.
# - The spring acts on the polar origins r, which differ from the area
#   centroids once modes n ≥ 2 are excited (products of modes create n = 1
#   harmonics in ∮ R³ e_φ dφ).
# - With line tension 0.05 the nuclei stay circular (⟨p⟩ = 3.545); they are
#   rarely touched by other bodies.
