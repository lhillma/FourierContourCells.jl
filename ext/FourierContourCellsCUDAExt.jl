module FourierContourCellsCUDAExt

# CUDA graphs for `run!`: one simulation step (`device_step!`) is captured once
# and replayed with one graph launch per step. The host synchronises once per
# chunk of steps to read the run status and the device clock. When the Verlet
# skin check raises the rebuild flag, the remaining steps of the chunk are
# frozen (see `isactive`), the host rebuilds the neighbour list eagerly and
# continues, so results are identical to `step!`.

using CUDA: CUDA, CUDABackend, CuMatrix, CuRef, cuBLAS
using FourierContourCells: Simulation, device_step!, update_neighbors!, to_host, STATUS_REBUILD
import FourierContourCells: run_steps!, matmul!
using LinearAlgebra: Transpose

## Allocation-free stacked products
#
# cuBLAS handles run in device pointer mode, and passing α, β as numbers
# allocates a device scalar per call. Such allocations make a captured graph
# unlaunchable a second time, so the scalars 1 and 0 are kept per context and
# element type.

const SCALARS = Dict{Tuple{CUDA.CuContext, DataType}, Tuple{Any, Any}}()

function scalars(::Type{T}) where {T}
    key = (CUDA.context(), T)
    one_zero = get(SCALARS, key, nothing)
    if one_zero === nothing
        CUDA.is_capturing() && error("cuBLAS scalars must be created outside graph capture")
        one_zero = SCALARS[key] = (CuRef{T}(one(T)), CuRef{T}(zero(T)))
    end
    return one_zero
end

const BLASFloat = Union{Float32, Float64}

function matmul!(C::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}) where {T <: BLASFloat}
    α, β = scalars(T)
    return cuBLAS.gemm!('N', 'N', α, A, B, β, C)
end
function matmul!(C::CuMatrix{T}, A::Transpose{T, <:CuMatrix{T}}, B::CuMatrix{T}) where {T <: BLASFloat}
    α, β = scalars(T)
    return cuBLAS.gemm!('T', 'N', α, parent(A), B, β, C)
end

## Graphs

# Captured step and what it is bound to: kernel arguments (array pointers, dt)
# are baked into the graph, so it is re-captured when the time step changes or
# the neighbour list matrix is reallocated (capacity growth).
mutable struct StepGraph
    exec::CUDA.CuGraphExec
    dt::Any
    idx::Any
    interval::Float64  # running estimate of the number of steps between rebuilds
end

const CHUNK_MIN = 4
const CHUNK_MAX = 256

# Steps per chunk: about a quarter of the rebuild interval, so that on average
# few frozen (wasted) steps follow a raised flag, and one sync covers many steps.
chunk_size(g::StepGraph, sim) = isempty(sim.contact_terms) ? CHUNK_MAX :
    clamp(round(Int, g.interval / 4), CHUNK_MIN, CHUNK_MAX)

function capture_step(sim::Simulation)
    graph = CUDA.capture(throw_error = false) do
        device_step!(sim)
    end
    if graph === nothing
        # a kernel was compiled during the capture: compile it outside (this
        # leaves the state untouched while the status is frozen) and retry
        CUDA.@sync begin
            fill!(sim.status, 1)
            device_step!(sim)
        end
        fill!(sim.status, 0)
        graph = CUDA.capture(() -> device_step!(sim))
    end
    return CUDA.instantiate(graph)
end

function step_graph!(sim::Simulation)
    g = sim.graph[]
    if !(g isa StepGraph && g.dt == sim.dt && g.idx === sim.neighbors.idx)
        interval = g isa StepGraph ? g.interval : 4.0 * CHUNK_MIN
        g = StepGraph(capture_step(sim), sim.dt, sim.neighbors.idx, interval)
        sim.graph[] = g
    end
    return g
end

function run_steps!(::CUDABackend, sim::Simulation, n::Integer)
    n > 0 || return sim
    # the captured step expects a valid neighbour list and a clear status
    isempty(sim.contact_terms) || update_neighbors!(sim; force = !sim.neighbors.valid)
    clock = to_host(sim.clock)[1]
    since_rebuild = 0
    done = 0
    while done < n
        g = step_graph!(sim)
        k = min(chunk_size(g, sim), n - done)
        for _ in 1:k
            CUDA.launch(g.exec)
        end
        status = to_host(sim.status)  # synchronises
        newclock = to_host(sim.clock)[1]
        executed = Int(newclock - clock)
        clock = newclock
        for _ in 1:executed
            sim.t += sim.dt
        end
        sim.step += executed
        done += executed
        since_rebuild += executed
        # a flag raised by the last step is left for the next step (as in step!)
        if status[STATUS_REBUILD] != 0 && done < n
            update_neighbors!(sim)  # sees the raised flag: rebuilds and clears it
            g.interval = (g.interval + since_rebuild) / 2
            since_rebuild = 0
        elseif since_rebuild > g.interval
            g.interval = min(since_rebuild, 4.0 * CHUNK_MAX)
        end
    end
    return sim
end

end
