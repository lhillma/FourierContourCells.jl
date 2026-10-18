module FourierContourCellsMakieExt

using FourierContourCells
using FourierContourCells: to_host
import FourierContourCells: cellview!, update_view!, cellplot, live_view, record_movie
using Makie

struct CellView{S, P}
    sim::S
    plot::P
    color::Any
    periodic::Bool
    types::Any          # nothing (all bodies) or a collection of body types
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

"""
Polygons of all bodies (plus periodic images) whose type is in `types`
(`nothing`: all), and the body index of each.
"""
function polygons(sim, periodic::Bool, types = nothing)
    X, Y = contours(sim)
    x = to_host(sim.bodies.x)
    y = to_host(sim.bodies.y)
    rmax = to_host(sim.geom.rmax)
    type = to_host(sim.bodies.type)
    Lx, Ly = sim.box.Lx, sim.box.Ly
    polys = Vector{Vector{Point2f}}()
    owner = Int[]
    sizehint!(polys, nbodies(sim))
    for i in 1:nbodies(sim)
        types === nothing || type[i] in types || continue
        base = [Point2f(X[k, i], Y[k, i]) for k in axes(X, 1)]
        push!(polys, base)
        push!(owner, i)
        periodic || continue
        sx = x[i] - rmax[i] < 0 ? Lx : x[i] + rmax[i] > Lx ? -Lx : zero(Lx)
        sy = y[i] - rmax[i] < 0 ? Ly : y[i] + rmax[i] > Ly ? -Ly : zero(Ly)
        for (dx, dy) in ((sx, zero(sy)), (zero(sx), sy), (sx, sy))
            (dx == 0 && dy == 0) && continue
            push!(polys, base .+ Point2f(dx, dy))
            push!(owner, i)
        end
    end
    return polys, owner
end

frame_title(sim) = "N = $(nbodies(sim)),  t = $(round(sim.t; digits = 2))  (step $(sim.step))"

function polygon_colors(sim, color, owner)
    v = body_values(sim, color)
    return v === nothing ? color : v[owner]
end

## API

function cellview!(ax, sim; color = :shape_index, colormap = nothing, colorrange = nothing,
        periodic::Bool = true, types = nothing, box::Bool = true, strokewidth = 0.5,
        strokecolor = :black, kwargs...)
    types = types isa Integer ? (types,) : types
    cmap, crange = default_colors(color)
    polys, owner = polygons(sim, periodic, types)
    plot = poly!(ax, polys; color = polygon_colors(sim, color, owner),
        colormap = something(colormap, cmap), colorrange = something(colorrange, crange),
        strokewidth, strokecolor, kwargs...)
    if box
        Lx, Ly = sim.box.Lx, sim.box.Ly
        lines!(ax, Point2f[(0, 0), (Lx, 0), (Lx, Ly), (0, Ly), (0, 0)]; color = :black, linewidth = 1)
        limits!(ax, 0, Lx, 0, Ly)
    end
    return CellView(sim, plot, color, periodic, types, Observable(frame_title(sim)))
end

function update_view!(v::CellView)
    polys, owner = polygons(v.sim, v.periodic, v.types)
    # update geometry and colours together, as the number of polygons may change
    Makie.update!(v.plot, polys; color = polygon_colors(v.sim, v.color, owner))
    v.title[] = frame_title(v.sim)
    return v
end

function update_view!(v::LayeredView)
    foreach(update_view!, v.views)
    v.title[] = first(v.views).title[]
    return v
end

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

function live_view(sim; steps_per_frame::Integer = 20, kwargs...)
    fig, view = cellplot(sim; kwargs...)
    controls = GridLayout(fig[2, 1:end]; tellwidth = false)
    running = Observable(true)
    button = Button(controls[1, 1]; label = @lift($running ? "pause" : "run"))
    on(_ -> running[] = !running[], button.clicks)
    Label(controls[1, 2], "steps / frame")
    slider = Slider(controls[1, 3]; range = 1:500, startvalue = steps_per_frame, width = 300)
    Label(controls[1, 4], @lift(string($(slider.value))))
    screen = display(fig)
    while isopen(screen)
        if running[]
            run!(sim, slider.value[])  # CUDA graphs on the GPU
            update_view!(view)
        end
        sleep(1 / 120)   # yield to the renderer
    end
    return sim
end

function record_movie(sim, path::AbstractString; frames::Integer = 300, steps_per_frame::Integer = 20,
        framerate::Integer = 30, kwargs...)
    fig, view = cellplot(sim; kwargs...)
    record(fig, path, 1:frames; framerate) do _
        run!(sim, steps_per_frame)
        update_view!(view)
    end
    return path
end

end
