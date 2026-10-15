"Perimeters ∮ √(R² + R'²) dφ of all bodies (on the compute backend)."
perimeters(sim::Simulation) = vec(sum(sqrt.(sim.geom.R .^ 2 .+ sim.geom.dR .^ 2); dims = 1)) .* sim.basis.Δφ

"Areas ½ ∮ R² dφ of all bodies (on the compute backend)."
areas(sim::Simulation) = vec(sum(sim.geom.R .^ 2; dims = 1)) .* (sim.basis.Δφ / 2)

"Shape index p = perimeter / √area of all bodies (2√π ≈ 3.545 for a circle)."
shape_index(sim::Simulation) = perimeters(sim) ./ sqrt.(areas(sim))

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
