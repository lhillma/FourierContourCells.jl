# Visualisation API. Methods are implemented in the Makie extension
# (`FourierContourCellsMakieExt`), loaded by `using GLMakie` (interactive and
# movies) or `using CairoMakie` (static figures and movies).

"""
    view = cellview!(ax, sim; color = :shape_index, colormap, colorrange,
                     periodic = true, types = nothing, box = true,
                     strokewidth = 0.5, strokecolor = :black)

Draw all body contours of `sim` into the Makie axis `ax` and return a view that
[`update_view!`](@ref) refreshes after the simulation advanced.

`color` is `:shape_index`, `:orientation`, `:type`, `:index`, a fixed colour
(e.g. `:gray`) or a function `sim -> values` with one value per body.
With `periodic = true`, bodies crossing the box boundary are also drawn at
their periodic images. `types` (a type or collection of types) draws only
bodies of those types; `box` draws the box outline and sets the limits.
"""
function cellview! end

"Redraw a view created by [`cellview!`](@ref) from the current simulation state."
function update_view! end

"""
    fig, view = cellplot(sim; size = (800, 700), colorbar = true, layers = nothing, kwargs...)

Figure with one axis showing the simulation box; `kwargs` go to [`cellview!`](@ref).

`layers` draws several views on top of each other, one per named tuple of
[`cellview!`](@ref) options (overriding `kwargs`), e.g. cells coloured by
shape index with their nuclei on top:

    cellplot(sim; layers = ((types = 1, color = :shape_index),
                            (types = 2, color = :steelblue)))

The colour bar belongs to the first layer with per-body colour values.
"""
function cellplot end

"""
    FourierContourCells.live_view(sim; steps_per_frame = 20, kwargs...)

Open an interactive window (GLMakie) that advances `sim` and redraws it until
the window is closed. Includes a run/pause button and a slider for the number
of steps per frame. `kwargs` go to [`cellplot`](@ref).
"""
function live_view end

"""
    FourierContourCells.record_movie(sim, path; frames = 300, steps_per_frame = 20, framerate = 30, kwargs...)

Advance `sim` and write one frame every `steps_per_frame` steps to the video
`path` (`.mp4`, `.mkv`, `.webm` or `.gif`). `kwargs` go to [`cellplot`](@ref).
"""
function record_movie end
