"""
    to_backend(backend, A::AbstractArray)

Copy `A` into a freshly allocated array on the KernelAbstractions `backend`.
Returns `A` itself if it already lives on that backend.
"""
function to_backend(backend, A::AbstractArray)
    get_backend(A) == backend && return A
    B = KernelAbstractions.allocate(backend, eltype(A), size(A))
    copyto!(B, A)
    return B
end

"Copy a (possibly device) array to host memory."
to_host(A::AbstractArray) = Array(A)

"Workgroup size used for all kernel launches."
const WORKGROUP = 256

"""
    KernelLayout

How the single-body (`BodyTerm`) kernel maps work on the `nθ` contour nodes
to work items:

- `BodyParallel()`: one work item per body, looping over its nodes. Best on
  the CPU, where a work item runs serially over a contiguous column.
- `NodeParallel()`: one work item per (node, body), with a workgroup of
  `nθ × B` bodies and node sums reduced in local memory. Best on GPUs, where
  per-body loops leave most threads idle and give uncoalesced memory access,
  but several times slower on the CPU (each node repeats the node sums).

The default is chosen per backend by [`default_layout`](@ref); pass
`layout` to [`Simulation`](@ref) to override it. The contact kernel is
node-parallel on all backends (on the CPU it is as fast or faster).
"""
abstract type KernelLayout end
struct BodyParallel <: KernelLayout end
struct NodeParallel <: KernelLayout end

"Default [`KernelLayout`](@ref) for a KernelAbstractions backend."
default_layout(::CPU) = BodyParallel()
default_layout(backend) = NodeParallel()

# Node-parallel workgroups hold all nθ nodes of a body; larger nθ falls back to
# the body-parallel kernels.
uses_node_layout(::NodeParallel, nθ) = nθ <= WORKGROUP
uses_node_layout(::BodyParallel, nθ) = false

# Device-side run state of a simulation (see `Simulation`): `status` is a
# small Int32 vector whose entry STATUS_REBUILD is set by the skin check when
# the neighbour list must be rebuilt. A set status freezes the remaining work
# of a step (heavy kernels and the integrator do nothing), which lets a CUDA
# graph run many steps and stop at the first one that needs a rebuild. Kernels
# also accept `status = nothing` (never frozen). `clock[1]` counts the steps
# on the device and is the Philox draw index of the noise.
const STATUS_REBUILD = 1
const STATUS_LENGTH = 2  # entry 2 reserved (neighbour list overflow, see notes)
@inline isactive(::Nothing) = true
Base.@propagate_inbounds isactive(status::AbstractVector) = status[STATUS_REBUILD] == 0
@inline draw_index(n::Integer) = n % UInt64
Base.@propagate_inbounds draw_index(clock::AbstractVector) = clock[1] % UInt64
@inline tick!(::Nothing, i) = nothing
Base.@propagate_inbounds function tick!(clock::AbstractVector, i)
    i == 1 && (clock[1] += one(eltype(clock)))
    return nothing
end

# Contour synthesis and its pullback as one stacked matrix product ([C; D; DD],
# see FourierBasis) or as three. One product saves two launches on GPUs; on
# the CPU, BLAS is faster with three products for these thin matrices.
stacked_products(::CPU) = false
stacked_products(backend) = true

# C = A·B for the stacked products. Backend extensions may override it, e.g.
# for an allocation-free cuBLAS call that can be captured in a CUDA graph.
matmul!(C, A, B) = mul!(C, A, B)

# Workgroup (nθ, B) and padded ndrange for node-parallel kernels. The body
# dimension is padded to a multiple of B, so every work item of a workgroup
# reaches @synchronize; kernels skip bodies i > N.
function node_launch(nθ, N)
    B = max(1, WORKGROUP ÷ nθ)
    return (nθ, B), (nθ, cld(N, B) * B)
end
