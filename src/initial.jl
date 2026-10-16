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

"""
    with_nuclei(cells; ratio = 0.5, type = 2, rng = Random.Xoshiro(2)) -> bodies

Append one round nucleus to every body of the host `Bodies` `cells`: radius
scale `ratio` times the cell's, centred on the cell, with uniformly random
orientation, body type `type` and the cell as parent. Bodies `1:N` are the
cells and `N+1:2N` the nuclei (nucleus `N+i` belongs to cell `i`). Bond them
with [`parent_bonds`](@ref); keep them inside with a negative `related`
coupling of [`Overlap`](@ref).
"""
function with_nuclei(cells::Bodies; ratio = 0.5, type::Integer = 2, rng = Random.Xoshiro(2))
    N = nbodies(cells)
    c = to_host(cells)
    coef = zeros(eltype(c.coef), size(c.coef))
    coef[1, :] .= 1
    nuclei = Bodies(copy(c.x), copy(c.y), 2π .* rand(rng, eltype(c.x), N), coef, ratio .* c.R0;
        type)
    nuclei.imgx .= c.imgx
    nuclei.imgy .= c.imgy
    bodies = vcat(c, nuclei)
    bodies.parent[(N + 1):(2N)] .= 1:N
    return bodies
end
