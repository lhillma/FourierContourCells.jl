module FourierContourCellsMakieExt

using FourierContourCells
using FourierContourCells: to_host
import FourierContourCells: cellview!, update_view!, cellplot, live_view, record_movie
using Makie

# One layer of bodies, drawn as a single mesh (filled contours) plus a single
# line plot (outlines) with fixed topology (see the Geometry section). Updating a
# frame only replaces vertex positions and colours.
struct CellView{S, P, L}
    sim::S
    plot::P             # the mesh (carries colormap and colorrange, e.g. for a Colorbar)
    outline::L
    vertices::Observable{Vector{Point2f}}
    colors::Observable{Any}
    strokes::Observable{Vector{Point2f}}
    color::Any
    periodic::Bool
    members::Vector{Int}  # drawn bodies (those of the selected types)
    title::Observable{String}
end

"Several `CellView`s of one simulation drawn on top of each other (`cellplot` layers)."
struct LayeredView{V}
    views::V
    title::Observable{String}
end

## Colours

body_values(sim, color::Symbol) =
    color === :shape_index ? Float32.(to_host(shape_index(sim))) :
    color === :orientation ? Float32.(mod.(to_host(sim.bodies.θ), 2π)) :
    color === :type ? Float32.(to_host(sim.bodies.type)) :
    color === :index ? Float32.(1:nbodies(sim)) : nothing
body_values(sim, f::Function) = Float32.(collect(f(sim)))
body_values(sim, color) = nothing

function default_colors(color)
    color === :shape_index && return (Makie.cgrad([:gray85, :firebrick]), (3.545f0, 4.2f0))
    color === :orientation && return (:hsv, (0.0f0, Float32(2π)))
    color === :type && return (:Set1_9, (1.0f0, 9.0f0))
    color === :index && return (:tab20, Makie.automatic)
    return (:viridis, Makie.automatic)
end

## Geometry
#
# Contours are star-shaped around the body centre (R(φ) in polar form), so
# each one is filled by a fixed triangle fan: vertex 0 at the centre, vertices
# 1…nθ at the contour nodes, triangles (0, k, k + 1). Each drawn body has
# `copies` slots: the body itself and, with `periodic`, up to three periodic
# images (shifted by Lx, Ly or both) for bodies crossing the box boundary.
# Unused image slots collapse onto the body centre (zero-area triangles,
# no outline), so the mesh topology never changes and Makie never has to
# triangulate.

copies(periodic) = periodic ? 4 : 1

"Triangle faces of `nslots` contour fans with `nθ` nodes each."
function fan_faces(nslots, nθ)
    faces = Makie.GLTriangleFace[]
    sizehint!(faces, nslots * nθ)
    for s in 0:(nslots - 1)
        c = s * (nθ + 1) + 1
        for k in 1:nθ
            push!(faces, Makie.GLTriangleFace(c, c + k, c + mod1(k + 1, nθ)))
        end
    end
    return faces
end

"""
Vertices (fans) and outline points (closed contours separated by NaN) of
the bodies `members`, and the body index of each vertex slot.
"""
function geometry(sim, periodic::Bool, members)
    X, Y = contours(sim)
    x = to_host(sim.bodies.x)
    y = to_host(sim.bodies.y)
    rmax = to_host(sim.geom.rmax)
    nθ = size(X, 1)
    Lx, Ly = sim.box.Lx, sim.box.Ly
    ncp = copies(periodic)
    verts = Vector{Point2f}(undef, length(members) * ncp * (nθ + 1))
    strokes = Vector{Point2f}(undef, length(members) * ncp * (nθ + 2))
    nan = Point2f(NaN, NaN)
    v = s = 0
    for i in members
        sx = x[i] - rmax[i] < 0 ? Lx : x[i] + rmax[i] > Lx ? -Lx : zero(Lx)
        sy = y[i] - rmax[i] < 0 ? Ly : y[i] + rmax[i] > Ly ? -Ly : zero(Ly)
        for c in 0:(ncp - 1)  # slot c: shift in x if c is odd, in y if c ≥ 2
            cx = isodd(c)
            cy = c >= 2
            used = (!cx || sx != 0) && (!cy || sy != 0)
            dx = cx ? sx : zero(sx)
            dy = cy ? sy : zero(sy)
            centre = Point2f(x[i] + dx, y[i] + dy)
            verts[v += 1] = centre
            for k in 1:nθ
                p = Point2f(X[k, i] + dx, Y[k, i] + dy)
                verts[v += 1] = used ? p : centre
                strokes[s += 1] = used ? p : nan
            end
            strokes[s += 1] = used ? Point2f(X[1, i] + dx, Y[1, i] + dy) : nan
            strokes[s += 1] = nan
        end
    end
    return verts, strokes
end

"Per-vertex colour values (or a fixed colour) for `geometry`."
function vertex_colors(sim, color, members, periodic::Bool)
    vals = body_values(sim, color)
    vals === nothing && return color
    n = copies(periodic) * (sim.basis.nθ + 1)
    return repeat(vals[members]; inner = n)
end

frame_title(sim) = "N = $(nbodies(sim)),  t = $(round(sim.t; digits = 2))  (step $(sim.step))"

## API

function cellview!(ax, sim; color = :shape_index, colormap = nothing, colorrange = nothing,
        periodic::Bool = true, types = nothing, box::Bool = true, strokewidth = 0.5,
        strokecolor = :black, kwargs...)
    types = types isa Integer ? (types,) : types
    cmap, crange = default_colors(color)
    type = to_host(sim.bodies.type)
    members = [i for i in 1:nbodies(sim) if types === nothing || type[i] in types]
    verts, strokes = geometry(sim, periodic, members)
    vertices = Observable(verts)
    colors = Observable{Any}(vertex_colors(sim, color, members, periodic))
    stroke_points = Observable(strokes)
    faces = fan_faces(length(members) * copies(periodic), sim.basis.nθ)
    plot = mesh!(ax, map(v -> Makie.GeometryBasics.Mesh(v, faces), vertices); color = colors,
        colormap = something(colormap, cmap), colorrange = something(colorrange, crange),
        shading = Makie.NoShading, kwargs...)
    outline = lines!(ax, stroke_points; color = strokecolor, linewidth = strokewidth)
    if box
        Lx, Ly = sim.box.Lx, sim.box.Ly
        lines!(ax, Point2f[(0, 0), (Lx, 0), (Lx, Ly), (0, Ly), (0, 0)]; color = :black, linewidth = 1)
        limits!(ax, 0, Lx, 0, Ly)
    end
    return CellView(sim, plot, outline, vertices, colors, stroke_points, color, periodic, members,
        Observable(frame_title(sim)))
end

# Updating a view has two parts: `frame_data` reads the simulation and builds
# everything to draw on the host (device transfers, vertices, colours, title),
# `apply!` hands it to Makie. live_view and record_movie run frame_data on a
# producer thread next to the simulation, so the main thread only renders.

function frame_data(v::CellView, sim = v.sim)
    verts, strokes = geometry(sim, v.periodic, v.members)
    return (; verts, strokes, color = vertex_colors(sim, v.color, v.members, v.periodic),
        title = frame_title(sim))
end
frame_data(v::LayeredView, sim = first(v.views).sim) = map(w -> frame_data(w, sim), v.views)

function apply!(v::CellView, data)
    # fixed topology: same vertex count every frame, only positions and colours change
    data.color isa AbstractVector && (v.colors[] = data.color)
    v.vertices[] = data.verts
    v.strokes[] = data.strokes
    v.title[] = data.title
    return v
end
function apply!(v::LayeredView, data)
    foreach(apply!, v.views, data)
    v.title[] = first(data).title
    return v
end

update_view!(v::Union{CellView, LayeredView}) = apply!(v, frame_data(v))

function cellplot(sim; size = (800, 700), colorbar::Bool = true, layers = nothing, kwargs...)
    fig = Figure(; size)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x", ylabel = "y")
    if layers === nothing
        view = cellview!(ax, sim; kwargs...)
        cbview = view
    else
        # later layers are drawn on top; the box is drawn once; layer options
        # override the common keyword arguments
        views = [cellview!(ax, sim; kwargs..., box = n == 1, layer...) for (n, layer) in enumerate(layers)]
        view = LayeredView(views, Observable(frame_title(sim)))
        k = findfirst(v -> body_values(sim, v.color) !== nothing, views)
        cbview = k === nothing ? first(views) : views[k]
    end
    onany(t -> ax.title = t, view.title; update = true)
    if colorbar && body_values(sim, cbview.color) !== nothing
        label = cbview.color isa Symbol ? replace(String(cbview.color), "_" => " ") : ""
        Colorbar(fig[1, 2], cbview.plot; label)
    end
    return fig, view
end

## Producer: simulation and frame preparation on another thread
#
# The producer advances the simulation by steps_per_frame steps, prepares the
# frame data and puts it into a channel of length 1, so it computes frame k + 1
# while the main thread renders (or encodes) frame k. `put!` blocks while the
# previous frame is not taken yet: every frame is exactly steps_per_frame
# steps after the previous one, as with sequential stepping. Only the producer
# touches the simulation until it has finished.

function start_producer(sim, view, steps::Function, nframes, active::Function)
    frames = Channel{Any}(1)
    task = Threads.@spawn begin
        try
            n = 0
            while n < nframes && isopen(frames)
                if active()
                    run!(sim, steps())  # CUDA graphs on the GPU
                    put!(frames, frame_data(view, sim))
                    n += 1
                else
                    sleep(0.01)
                end
            end
        catch err
            # closing the channel (window closed) ends the producer
            err isa InvalidStateException && !isopen(frames) || rethrow()
        finally
            close(frames)
        end
    end
    return frames, task
end

function live_view(sim; steps_per_frame::Integer = 20, kwargs...)
    fig, view = cellplot(sim; kwargs...)
    controls = GridLayout(fig[2, 1:end]; tellwidth = false)
    running = Observable(true)
    button = Button(controls[1, 1]; label = @lift($running ? "pause" : "run"))
    on(_ -> running[] = !running[], button.clicks)
    Label(controls[1, 2], "steps / frame")
    slider = Slider(controls[1, 3]; range = 1:500, startvalue = steps_per_frame, width = 300)
    Label(controls[1, 4], @lift(string($(slider.value))))
    # the producer reads the controls through atomics, not Observables
    active = Threads.Atomic{Bool}(true)
    steps = Threads.Atomic{Int}(steps_per_frame)
    on(r -> active[] = r, running)
    on(n -> steps[] = n, slider.value)
    screen = display(fig)
    frames, task = start_producer(sim, view, () -> steps[], typemax(Int), () -> active[])
    try
        while isopen(screen) && !istaskdone(task)
            isready(frames) && apply!(view, take!(frames))
            sleep(1 / 120)   # yield to the renderer
        end
    finally
        close(frames)
        wait(task)       # rethrows errors of the producer; afterwards sim is ours again
    end
    return sim
end

function record_movie(sim, path::AbstractString; frames::Integer = 300, steps_per_frame::Integer = 20,
        framerate::Integer = 30, kwargs...)
    fig, view = cellplot(sim; kwargs...)
    channel, task = start_producer(sim, view, () -> steps_per_frame, frames, () -> true)
    try
        record(fig, path, 1:frames; framerate) do _
            apply!(view, take!(channel))
        end
    finally
        close(channel)
        wait(task)
    end
    return path
end

end
