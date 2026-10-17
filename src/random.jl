# Counter-based random numbers: Philox4x32-10 of
#
#   J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random
#   numbers: as easy as 1, 2, 3", SC '11 (2011), doi:10.1145/2063384.2063405.
#
# A Philox draw is a pure function of a 128-bit counter and a 64-bit key, so
# every work item can generate its own numbers without carried state, and the
# stream is identical on every backend.

const PHILOX_M0 = 0xD2511F53
const PHILOX_M1 = 0xCD9E8D57
const PHILOX_W0 = 0x9E3779B9
const PHILOX_W1 = 0xBB67AE85

@inline function _mulhilo(a::UInt32, b::UInt32)
    p = UInt64(a) * UInt64(b)
    return (p >> 32) % UInt32, p % UInt32
end

"""
    philox4x32(ctr::NTuple{4, UInt32}, key::NTuple{2, UInt32}) -> NTuple{4, UInt32}

Philox4x32 with 10 rounds (Salmon et al., SC '11, doi:10.1145/2063384.2063405).
"""
@inline function philox4x32(ctr::NTuple{4, UInt32}, key::NTuple{2, UInt32})
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in 1:10
        hi0, lo0 = _mulhilo(PHILOX_M0, c0)
        hi1, lo1 = _mulhilo(PHILOX_M1, c2)
        c0, c1, c2, c3 = hi1 ⊻ c1 ⊻ k0, lo1, hi0 ⊻ c3 ⊻ k1, lo0
        k0 += PHILOX_W0
        k1 += PHILOX_W1
    end
    return (c0, c1, c2, c3)
end

# Uniform numbers in the open interval (0, 1), (j + ½)·2⁻ᵇ for random b-bit j:
# b = 23 for Float32, 52 (from two words) for Float64, so that j + ½ is exact
# (with one more bit, the largest value rounds up to 1). Float32 constants
# avoid silent promotion to Float64, which is slow on consumer GPUs.
@inline uniform_open(::Type{Float32}, a::UInt32, b::UInt32) = (Float32(a >> 9) + 0.5f0) * Float32(0x1p-23)
@inline uniform_open(::Type{Float64}, a::UInt32, b::UInt32) =
    (Float64(((UInt64(a) << 32) | b) >> 12) + 0.5) * 0x1p-52

# Box–Muller: two independent standard normals from two uniforms in (0, 1).
@inline function box_muller(u1::T, u2::T) where {T}
    r = sqrt(-2 * log(u1))
    s, c = sincos(2 * T(π) * u2)
    return r * c, r * s
end

"""
    philox_normals(T, key, i, n) -> NTuple{4, T}

Four independent standard normal numbers for body `i` in draw `n` (e.g. the
step number), from two Philox blocks with counters (i, n_lo, n_hi, 0|1).
"""
@inline function philox_normals(::Type{T}, key::NTuple{2, UInt32}, i, n) where {T}
    lo = n % UInt32
    hi = (n >> 32) % UInt32
    a = philox4x32((i % UInt32, lo, hi, UInt32(0)), key)
    b = philox4x32((i % UInt32, lo, hi, UInt32(1)), key)
    z1, z2 = box_muller(uniform_open(T, a[1], a[2]), uniform_open(T, a[3], a[4]))
    z3, z4 = box_muller(uniform_open(T, b[1], b[2]), uniform_open(T, b[3], b[4]))
    return (z1, z2, z3, z4)
end

"Philox key (two 32-bit words) from an integer seed."
philox_key(seed::Integer) = ((UInt64(seed) % UInt32), ((UInt64(seed) >> 32) % UInt32))

# Standard normal noise for one step. One work item per body i.
#
#   ξ     (out) 3 × N standard normal numbers of body i in column i (rows: x,
#         y, θ), the first three of philox_normals(T, key, i, n)
#   key   Philox key (from the simulation seed)
#   n     draw index (the step number), so every step gets fresh numbers
@kernel function noise_kernel!(ξ, key, n)
    i = @index(Global)
    z = philox_normals(eltype(ξ), key, i, n)
    ξ[1, i] = z[1]
    ξ[2, i] = z[2]
    ξ[3, i] = z[3]
end

"Fill `ξ` (3 × N) with the standard normal noise of draw `n`."
function philox_noise!(ξ, key, n::Integer)
    noise_kernel!(get_backend(ξ), WORKGROUP)(ξ, key, UInt64(n); ndrange = size(ξ, 2))
    return ξ
end
