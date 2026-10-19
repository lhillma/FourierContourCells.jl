# Cells with a nucleus.
#
#   julia -t auto --project=examples examples/nucleus.jl          # live window
#   julia -t auto --project=examples examples/nucleus.jl check    # headless run + diagnostics
#        options: drag=parent|substrate  Jw=<own-nucleus wrap coupling>  k=<bond constant>
#                 ηn=<nucleus line tension>
#   julia -t auto --project=examples examples/nucleus.jl movie    # writes output/nucleus.mp4
#   julia -t auto --project=examples examples/nucleus.jl test     # pull-in (wetting) check
#
# Runs on a CUDA GPU in Float32 if one is available, else on the CPU in Float64
# (see backend.jl; options for live, check and movie: backend=cpu|gpu
# T=Float32|Float64). The `test` mode always runs on the CPU in Float64.
#
# Extension of the Fourier contour cell model of N. Saito and S. Ishihara,
# Sci. Adv. 10, eadi8433 (2024), doi:10.1126/sciadv.adi8433.
#
# Model. Every cell (type 1) contains one nucleus (type 2), a Fourier contour
# body of its own whose `parent` is the cell (`with_nuclei`).
#
#   - Overlap: overlap with any other cell or nucleus costs J > 0 per unit
#     area. A cell and its own nucleus gain -Jw per unit overlap area
#     (negative `related` coupling): the membrane wraps the nucleus, and parts
#     of the nucleus poking out are pulled back. This does not cause wetting:
#     the overlap is at most the (fixed) nucleus area, and a nucleus fully
#     inside its cell feels no force from it.
#   - Activity: the nucleus has v0 = 0. Its orientation θ diffuses
#     independently of the cell's (rotational noise Dr) but only sets the frame
#     of its shape modes.
#   - Anchoring: a harmonic bond E = (k/2) |r_n - r_c|² between the centres of
#     nucleus and cell (`HarmonicBond`, `parent_bonds`), a crude model of the
#     cytoskeleton.
#   - Drag: the nucleus has drag against the cytoplasm, i.e. relative to its
#     cell (`BodyType(drag = :parent)`): the cell bears the nucleus's forces
#     and carries it along, so the bond is internal and only re-centres the
#     nucleus at rate μr_n k. Substrate drag on the nucleus would double-count
#     the friction of the cell's footprint (an isolated cell would move at
#     v0/2).
#   - Nucleus stiffness: a stronger line tension on type 2 than on type 1.

using FourierContourCells
using GLMakie
include("backend.jl")

const CELL, NUCLEUS = 1, 2

# ---------------------------------------------------------------------------
# Parameters: "soft fluid" cells of Saito & Ishihara (2024), Fig. 2C

const WRAP = 2.0    # Jw, own-nucleus wrap coupling (> J: wins against neighbours)
const BOND = 0.2    # k, nucleus–cell bond constant

# Overlap couplings indexed (type_i, type_j); same-parent pairs use `related`.
function energies(; Jw = WRAP, k = BOND, ηn = 0.05)
    J = [1.0 1.0; 1.0 1.0]
    Jrelated = [0.0 -Jw; -Jw 0.0]
    return (OnTypes(Tension(0.01), CELL), OnTypes(Tension(ηn), NUCLEUS),
        Overlap(J; related = Jrelated), HarmonicBond(k))
end

cell_type(; v0 = 0.06, Dr = 0.01) = BodyType(; v0, Dr, μab = 0.1)
nucleus_type(; Dr = 0.01, drag = :parent) = BodyType(; v0 = 0.0, Dr, μab = 0.1, drag)

function monolayer(; drag = :parent, kwargs...)
    cells, box = hexagonal_monolayer(32, 32; packing = 0.95)
    bodies = with_nuclei(cells; ratio = 0.5, type = NUCLEUS)
    backend, T = backend_and_precision()
    return Simulation(bodies, box; types = (cell_type(), nucleus_type(; drag)),
        energies = energies(; kwargs...), bonds = parent_bonds(bodies), dt = 0.005, backend, T)
end

# cells coloured by shape index, nuclei on top
const LAYERS = ((types = CELL, color = :shape_index), (types = NUCLEUS, color = :steelblue))

# ---------------------------------------------------------------------------
# Diagnostics

mean(v) = sum(v) / length(v)

"Centre offset |r_n - r_c| of every nucleus from its cell (host)."
function nucleus_offsets(sim)
    b = FourierContourCells.to_host(sim.bodies)
    mi = FourierContourCells.minimum_image
    return [hypot(mi(b.x[n] - b.x[b.parent[n]], sim.box.Lx), mi(b.y[n] - b.y[b.parent[n]], sim.box.Ly))
            for n in findall(==(NUCLEUS), b.type)]
end

function report(sim)
    d = nucleus_offsets(sim)
    pr = protrusion(sim; type = NUCLEUS)
    println("t = ", round(sim.t; digits = 1),
        "  offset mean/max = ", round(mean(d); sigdigits = 2), "/", round(maximum(d); digits = 3),
        "  protruding nuclei = ", count(>(0), pr), " (max ", round(maximum(pr); digits = 3), ")",
        "  ⟨p⟩ cells/nuclei = ", round(mean(Array(shape_index(sim; type = CELL))); digits = 3), "/",
        round(mean(Array(shape_index(sim; type = NUCLEUS))); digits = 3))
end

"""
Shape statistics per body type, to detect artefacts such as wetting or
excited high modes: mean amplitude |cₙ| = √(aₙ² + bₙ²) of each mode n, mean
and maximum shape index, and the area range relative to πR0².
"""
function shape_report(sim)
    coef = Array(sim.bodies.coef)
    typ = Array(sim.bodies.type)
    K = nmodes(sim.basis)
    for (t, name) in ((CELL, "cells "), (NUCLEUS, "nuclei"))
        sel = typ .== t
        amp = [mean(hypot.(coef[1 + m, sel], coef[1 + K + m, sel])) for m in 1:K]
        p = Array(shape_index(sim; type = t))
        A = Array(areas(sim; type = t)) ./ (π .* Array(sim.bodies.R0)[sel] .^ 2)
        println("  ", name, "  ⟨|cₙ|⟩ n = 2…", K + 1, ": ", join(round.(amp; sigdigits = 2), " "),
            "   p mean/max = ", round(mean(p); digits = 3), "/", round(maximum(p); digits = 3),
            "   A/πR0² ∈ [", round(minimum(A); digits = 4), ", ", round(maximum(A); digits = 4), "]")
    end
end

"Mean cell displacement |Δr| since the unwrapped positions `X0`."
function mean_cell_displacement(sim, X0)
    Δ = (unwrapped_positions(sim) .- X0)[:, Array(sim.bodies.type) .== CELL]
    return mean(vec(sqrt.(sum(abs2, Δ; dims = 1))))
end

# ---------------------------------------------------------------------------
# Pull-in check: a soft nucleus starting half outside its cell, without bond.
# The wrap coupling must pull it in without wetting artefacts (excited modes,
# irregular shape); without it the nucleus stays where it is.

function check(name, value, expected; atol)
    ok = isapprox(value, expected; atol)
    println(ok ? "  pass  " : "  FAIL  ", rpad(name, 44), "got ", value, ", expected ", expected)
    return ok
end

function pull_in(; Jw, t = 50.0)
    cell = Bodies([10.0], [10.0], [0.0], [1.0; zeros(10);;], [1.0])
    bodies = with_nuclei(cell; type = NUCLEUS)
    bodies.x[2] += 0.7          # 0.2 R₀ outside the cell contour
    sim = Simulation(bodies, PeriodicBox(20.0, 20.0); types = (cell_type(; v0 = 0.0, Dr = 0.0),
        nucleus_type(; Dr = 0.0)), energies = energies(; Jw, k = 0.0, ηn = 0.005), dt = 0.005)
    step!(sim, round(Int, t / sim.dt))
    return sim
end

function run_tests()
    ok = true
    println("pull-in, Jw = 0: soft nucleus starting 0.2 R₀ outside, no bond")
    ok &= check("protrusion", protrusion(pull_in(; Jw = 0.0))[2], 0.2; atol = 1e-12)
    println("pull-in, Jw = $WRAP")
    sim = pull_in(; Jw = WRAP)
    K = nmodes(sim.basis)
    c = Array(sim.bodies.coef)
    ok &= check("protrusion (positive part)", max(protrusion(sim)[2], 0.0), 0.0; atol = 1e-3)
    ok &= check("largest nucleus mode amplitude", maximum(hypot(c[1 + q, 2], c[1 + K + q, 2]) for q in 1:K),
        0.0; atol = 0.01)
    ok &= check("nucleus shape index", Array(shape_index(sim))[2], 2sqrt(π); atol = 1e-3)
    ok || error("nucleus checks failed")
    return nothing
end

# ---------------------------------------------------------------------------

mode = get(ARGS, 1, "live")
if mode == "test"
    run_tests()
elseif mode == "check"
    drag = Symbol(get(OPTS, "drag", "parent"))
    Jw = parse(Float64, get(OPTS, "Jw", string(WRAP)))
    k = parse(Float64, get(OPTS, "k", string(BOND)))
    ηn = parse(Float64, get(OPTS, "ηn", "0.05"))
    println("drag = ", drag, ", Jw = ", Jw, ", k = ", k, ", ηn = ", ηn)
    sim = monolayer(; drag, Jw, k, ηn)
    println(sim)
    X0 = unwrapped_positions(sim)
    run!(sim, 100)  # compile (and capture the CUDA graph) outside the timing
    seconds = 0.0
    for _ in 1:10
        global seconds += @elapsed begin
            run!(sim, 2_000)
            CUDA.functional() && CUDA.synchronize()
        end
        report(sim)
    end
    println("time per step: ", round(seconds / 20_000 * 1e3; digits = 3), " ms")
    shape_report(sim)
    println("mean cell displacement: ", round(mean_cell_displacement(sim, X0); digits = 3))
    mkpath(joinpath(@__DIR__, "..", "output"))
    path = joinpath(@__DIR__, "..", "output", "nucleus_$(drag)_Jw$(Jw)_k$(k)_eta$(ηn).png")
    save(path, first(cellplot(sim; layers = LAYERS)))
    println("wrote ", path)
elseif mode == "movie"
    sim = monolayer()
    run!(sim, 20_000)
    mkpath(joinpath(@__DIR__, "..", "output"))
    path = joinpath(@__DIR__, "..", "output", "nucleus.mp4")
    println("wrote ", FourierContourCells.record_movie(sim, path; frames = 600, steps_per_frame = 100,
        layers = LAYERS))
elseif mode == "live"
    FourierContourCells.live_view(monolayer(); steps_per_frame = 50, layers = LAYERS)
end

# ---------------------------------------------------------------------------
# Notes from this prototype (extensibility test of the library)
#
# The first version of this script needed its own kernels, a hand-copied
# `step!`, rewritten viewer loops and a colour-map trick. Library additions
# that came out of it: `BodyType(drag = :parent)`, `BondTerm`/`HarmonicBond`
# with `Simulation(...; bonds)` and `parent_bonds`, the prototyping
# `hook!` keyword, `with_nuclei`, `vcat(::Bodies...)`, `protrusion`, `type =`
# filters on observables, plot `layers`, and public kernel helpers.
#
# Remaining friction:
#
# - Cost: 1024 cells + 1024 nuclei take 2.2× the time of the cells alone
#   (9.0 vs 4.0 ms/step on 1 thread, 3.5 vs 1.6 ms/step on 16 threads).
#   Nuclei enter the neighbour lists of nearby cells, and the contact kernel
#   evaluates each cell–own-nucleus pair (needed for the wrap coupling).
#   After the GPU work (`check`, t = 100): 2.3 ms/step on 16 CPU threads
#   (Float64), 0.26 ms/step on an RTX A2000 Laptop GPU (Float32, CUDA
#   graphs). GPU and CPU statistics agree (mean cell displacement 0.93 vs
#   0.94, ⟨p⟩ of cells 3.819 vs 3.823, no protruding nuclei).
# - All body types share the number of shape modes and contour nodes.
# - Bond energies are not included in `body_energy`.
#
# Modelling observations (to discuss):
#
# - Confinement. With `related` = 0 the membrane does not feel its own
#   nucleus, so elongating cells narrow past it: 6–7 nuclei protrude at
#   t = 100 (by up to 0.027 R₀). The wrap coupling Jw (negative `related`
#   overlap) fixes this at no extra cost. Monolayer at t = 100 (relative
#   drag; protruding nuclei, nucleus ⟨|cₙ|⟩ for n = 2 and 6, max nucleus p):
#     Jw = 0, k = 1                  7   7e-5 / 1e-5   3.551
#     Jw = 2, k = 1                  1   7e-5 / 1e-5   3.549   (0.001 R₀)
#     Jw = 0, k = 0.2                6   3e-5 / 6e-6   3.546
#     Jw = 2, k = 0.2                0   3e-5 / 5e-6   3.545
#     Jw = 2, k = 0.2, ηn = 0.005    0   4e-5 / 2e-5   3.546
#     Jw = 5, k = 0.2, ηn = 0.005    0   5e-5 / 2e-5   3.547
#   Cell shapes and displacements are unchanged within noise. No wetting:
#   a soft nucleus started 0.2 R₀ outside its cell (no spring) is pulled in
#   by t = 2. Its modes stay ≤ 0.005 and fall off with n. The cell bulges
#   briefly (|c₂| ≈ 0.014) to wrap the nucleus, and line tension relaxes the
#   bulge again. Once inside, the nucleus does not stick to the membrane. The
#   alternative (an energy for nucleus area outside its cell) would need an
#   interaction range of about a nucleus diameter for all pairs.
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
#   harmonics in ∮ R³ e_φ dφ). Accepted for now.
# - With line tension 0.05 the nuclei stay circular (⟨p⟩ = 3.545); they are
#   rarely touched by other bodies.
