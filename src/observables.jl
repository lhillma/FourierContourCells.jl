# Per-body observables take an optional `type`: values of bodies of that type
# only (in body order), otherwise of all bodies.
select_type(v, sim, ::Nothing) = v
select_type(v, sim, type::Integer) = v[sim.bodies.type .== type]

"Perimeters ∮ √(R² + R'²) dφ (on the compute backend); `type` selects one body type."
perimeters(sim::Simulation; type = nothing) =
    select_type(vec(sum(sqrt.(sim.geom.R .^ 2 .+ sim.geom.dR .^ 2); dims = 1)) .* sim.basis.Δφ, sim, type)

"Areas ½ ∮ R² dφ (on the compute backend); `type` selects one body type."
areas(sim::Simulation; type = nothing) =
    select_type(vec(sum(sim.geom.R .^ 2; dims = 1)) .* (sim.basis.Δφ / 2), sim, type)

"""
Shape index p = perimeter / √area (2√π ≈ 3.545 for a circle); `type` selects
one body type.
"""
shape_index(sim::Simulation; type = nothing) = perimeters(sim; type) ./ sqrt.(areas(sim; type))

"Wrapped centre positions as a `2 × N` host matrix."
positions(sim::Simulation) = permutedims(hcat(to_host(sim.bodies.x), to_host(sim.bodies.y)))

"Unwrapped centre positions (periodic images resolved) as a `2 × N` host matrix."
function unwrapped_positions(sim::Simulation)
    b = sim.bodies
    x = to_host(b.x) .+ to_host(b.imgx) .* sim.box.Lx
    y = to_host(b.y) .+ to_host(b.imgy) .* sim.box.Ly
    return permutedims(hcat(x, y))
end

"""
    contours(sim) -> (X, Y)

Host `nθ × N` matrices of contour node coordinates around the wrapped centres.
"""
function contours(sim::Simulation)
    R = to_host(sim.geom.R)
    x = to_host(sim.bodies.x)
    y = to_host(sim.bodies.y)
    φ = [node_angle(sim.basis, k) for k in 1:(sim.basis.nθ)]
    X = x' .+ R .* cos.(φ)
    Y = y' .+ R .* sin.(φ)
    return X, Y
end

"Total single-body (e.g. tension) energy of the last gradient evaluation."
body_energy(sim::Simulation) = sum(sim.grad.Ebody)

"""
    protrusion(sim; type = nothing) -> Vector

How far each body pokes out of its parent: the largest radial distance of
its contour nodes u beyond the parent's contour, maxₖ (|uₖ - r_c| - R_c(ψₖ))
with ψₖ the angle of uₖ seen from the parent centre r_c. Negative: the body
lies inside its parent (by at least that margin along the parent's rays).
`NaN` for bodies that are their own parent. Host computation, for
diagnostics.
"""
function protrusion(sim::Simulation; type = nothing)
    b = to_host(sim.bodies)
    R = to_host(sim.geom.R)
    lab = to_host(sim.geom.lab)
    K = nmodes(sim.basis)
    Lx, Ly = sim.box.Lx, sim.box.Ly
    out = fill(convert(eltype(R), NaN), nbodies(b))
    for i in 1:nbodies(b)
        c = b.parent[i]
        c == i && continue
        p = typemin(eltype(R))
        for k in 1:(sim.basis.nθ)
            s, co = sincos(node_angle(sim.basis, k))
            dx = minimum_image(b.x[i] + R[k, i] * co - b.x[c], Lx)
            dy = minimum_image(b.y[i] + R[k, i] * s - b.y[c], Ly)
            p = max(p, hypot(dx, dy) - contour_radius(lab, c, b.R0[c], K, atan(dy, dx)))
        end
        out[i] = p
    end
    return type === nothing ? out : out[b.type .== type]
end
