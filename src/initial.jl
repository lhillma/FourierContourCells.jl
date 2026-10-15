"""
    hexagonal_monolayer(nx, ny; packing = 0.95, R0 = 1.0, maxmode = 6,
                        rng = Random.Xoshiro(1)) -> (bodies, box)

Circular cells of radius `R0` on a hexagonal lattice with `nx` cells per row
and `ny` rows, at area fraction `packing`, with uniformly random orientations.
The box has aspect ratio Lx : Ly = 2nx : √3 ny (2 : √3 for nx = ny, as in
Saito & Ishihara (2024), doi:10.1126/sciadv.adi8433); `ny` must be even for
a periodic lattice. The lattice construction follows their reference code.
"""
function hexagonal_monolayer(nx::Integer, ny::Integer; packing = 0.95, R0 = 1.0, maxmode::Integer = 6,
        rng = Random.Xoshiro(1))
    iseven(ny) || throw(ArgumentError("ny must be even for a periodic hexagonal lattice"))
    N = nx * ny
    # half the lattice spacing, from N π R0² = packing · Lx Ly with Lx Ly = 2√3 r² N
    r = R0 * sqrt(π / (2sqrt(3) * packing))
    box = PeriodicBox(2r * nx, sqrt(3) * r * ny)
    x = zeros(N)
    y = zeros(N)
    for n in 0:(N - 1)
        row, col = divrem(n, nx)
        x[n + 1] = (2col + isodd(row)) * r
        y[n + 1] = sqrt(3) * r * row
    end
    θ = 2π .* rand(rng, N)
    coef = zeros(2maxmode - 1, N)
    coef[1, :] .= 1
    return Bodies(x, y, θ, coef, fill(float(R0), N)), box
end
