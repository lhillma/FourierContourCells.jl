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
