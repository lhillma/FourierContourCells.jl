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
# Model (the choices and the measurements behind them: docs/nucleus.md).
# Every cell (type 1) contains one nucleus (type 2), a Fourier contour body
# of its own whose `parent` is the cell (`with_nuclei`).
#
#   - Overlap: overlap with any other cell or nucleus costs J = 1 per unit
#     area; a cell and its own nucleus gain Jw per unit overlap area
#     (negative `related` coupling), so the membrane wraps the nucleus.
#   - Activity: the nucleus has v0 = 0; its orientation only sets the frame
#     of its shape modes.
#   - Anchoring: a harmonic bond (k/2) |r_n - r_c|² between the centres of
#     nucleus and cell (`HarmonicBond`, `parent_bonds`).
#   - Drag: the nucleus has drag relative to its cell (`drag = :parent`).
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
# Pull-in check: a soft nucleus poking 0.2 R₀ out of its cell, without bond.
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
