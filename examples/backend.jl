# Compute backend and precision of the examples, included by them: a CUDA GPU
# in Float32 if one is available (Float64 is slow on consumer GPUs), otherwise
# the CPU in Float64. Command line options override the choice:
#
#   backend=cpu|gpu  T=Float32|Float64
#
# On the GPU, `run!` replays steps as CUDA graphs (see the README).

using FourierContourCells
using CUDA

# Command line options key=value
const OPTS = Dict(Pair(split(a, "="; limit = 2)...) for a in ARGS if occursin('=', a))

function backend_and_precision()
    gpu = get(OPTS, "backend", CUDA.functional() ? "gpu" : "cpu") == "gpu"
    gpu && !CUDA.functional() && error("backend=gpu, but CUDA is not functional")
    backend = gpu ? CUDABackend() : CPU()
    T = Dict("Float32" => Float32, "Float64" => Float64)[get(OPTS, "T", gpu ? "Float32" : "Float64")]
    return backend, T
end
