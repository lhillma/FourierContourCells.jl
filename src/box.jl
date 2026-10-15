"""
    PeriodicBox(Lx, Ly)

Rectangular simulation box `[0, Lx) × [0, Ly)` with periodic boundaries.
"""
struct PeriodicBox{T}
    Lx::T
    Ly::T
end

PeriodicBox{T}(b::PeriodicBox) where {T} = PeriodicBox{T}(b.Lx, b.Ly)

"Minimum-image convention for a displacement `d` in a periodic dimension of length `L`."
@inline minimum_image(d, L) = d - L * round(d / L)

"""
    wrap(x, L) -> (x′, shift)

Map `x` into `[0, L)` and return the number of periods `shift` subtracted.
"""
@inline function wrap(x, L)
    s = floor(x / L)
    x′ = x - s * L
    # guard against x′ == L from rounding when x is a tiny negative number
    if x′ >= L
        x′ = zero(x′)
        s += one(s)
    end
    return x′, unsafe_trunc(Int32, s)
end

area(b::PeriodicBox) = b.Lx * b.Ly
