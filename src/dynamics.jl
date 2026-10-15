# Overdamped Langevin dynamics, integrated with the Euler–Maruyama scheme:
#
#   ṙ  = v0 e(θ) - μr ∂E/∂r + √(2Dt) ξ
#   θ̇  = -μθ ∂E/∂θ + √(2Dr) ξ
#   ȧₙ = -μab ∂E/∂aₙ,  ḃₙ = -μab ∂E/∂bₙ

"""
Bound on s = Σₙ √(aₙ² + bₙ²) keeping R(φ) > 0 under the hard area constraint.

Derivation: |aₙ cos + bₙ sin| ≤ √(aₙ² + bₙ²), so R/R₀ ≥ a₀ - s. With
q = Σₙ (aₙ² + bₙ²) ≤ s², the area constraint gives a₀ = √(1 - q/2) ≥ √(1 - s²/2),
which exceeds s whenever s² < 2/3.
"""
const SHAPE_BOUND = sqrt(2 / 3)

"""
    enforce_shape!(coef, i, K)

Rescale the shape modes of body `i` (column `i` of `coef`, `K` modes) so that
Σₙ √(aₙ² + bₙ²) ≤ √(2/3) (see [`SHAPE_BOUND`](@ref)) and set
a₀ = √(1 - Σₙ (aₙ² + bₙ²)/2). The latter is the hard area constraint: by
orthogonality of the Fourier modes, the area is
½ ∮ R² dφ = π R₀² (a₀² + Σₙ (aₙ² + bₙ²)/2), which equals πR₀² exactly then.
Callable from kernels.

Note: the reference implementation (FourierContourCell.cpp, `setR`) only
rescales the n = 2 mode because of a loop bug; here all modes are rescaled as
described in the paper.
"""
@inline function enforce_shape!(coef, i, K)
    T = eltype(coef)
    s = zero(T)
    for m in 1:K
        s += sqrt(coef[1 + m, i]^2 + coef[1 + K + m, i]^2)
    end
    bound = T(SHAPE_BOUND)
    if s > bound
        f = bound / s
        for m in 1:K
            coef[1 + m, i] *= f
            coef[1 + K + m, i] *= f
        end
    end
    q = zero(T)
    for m in 1:K
        q += coef[1 + m, i]^2 + coef[1 + K + m, i]^2
    end
    coef[1, i] = sqrt(1 - q / 2)
    return nothing
end

# Apply `enforce_shape!` to every body. One work item per body i.
#   coef  (in/out) ncoef × N body-frame coefficients;  K  number of shape modes
@kernel function enforce_shape_kernel!(coef, K)
    i = @index(Global)
    enforce_shape!(coef, i, K)
end

function enforce_shape!(bodies::Bodies, basis::FourierBasis)
    enforce_shape_kernel!(get_backend(bodies.coef), WORKGROUP)(bodies.coef, nmodes(basis);
        ndrange = nbodies(bodies))
    return bodies
end

# One Euler–Maruyama step for every body. One work item per body i.
#
# With p = types[type[i]] the parameters of body i's type and ξ standard normal
# numbers, the update reads
#   r  ← r + dt (v0 (cos θ, sin θ) - μr ∂E/∂r) + √(2 Dt dt) (ξ₁, ξ₂)
#   θ  ← θ - dt μθ ∂E/∂θ + √(2 Dr dt) ξ₃
#   aₙ ← aₙ - dt μab ∂E/∂aₙ,  bₙ ← bₙ - dt μab ∂E/∂bₙ
# using the orientation θ before the update for the propulsion (explicit
# scheme). Afterwards the shape constraints are enforced and the position is
# wrapped into the box, counting crossed periods in the image counters.
#
#   x, y, θ      (in/out) N, positions and orientations
#   imgx, imgy   (in/out) N, periodic image counters
#   coef         (in/out) ncoef × N body-frame coefficients
#   type         N, body types (index into `types`)
#   gx, gy, gθ   N, energy gradients ∂E/∂x, ∂E/∂y, ∂E/∂θ
#   gcoef        ncoef × N, ∂E/∂[a₀; aₙ; bₙ] (row 1 unused)
#   ξ            3 × N standard normal random numbers
#   types        tuple of BodyType (isbits, indexed with the type)
#   dt           time step; Lx, Ly: box size; K: number of shape modes
@kernel function euler_maruyama_kernel!(x, y, θ, imgx, imgy, coef, @Const(type), @Const(gx),
        @Const(gy), @Const(gθ), @Const(gcoef), @Const(ξ), types, dt, Lx, Ly, K)
    i = @index(Global)
    p = types[type[i]]
    s, c = sincos(θ[i])
    σt = sqrt(2 * p.Dt * dt)
    xn = x[i] + dt * (p.v0 * c - p.μr * gx[i]) + σt * ξ[1, i]
    yn = y[i] + dt * (p.v0 * s - p.μr * gy[i]) + σt * ξ[2, i]
    θ[i] += -dt * p.μθ * gθ[i] + sqrt(2 * p.Dr * dt) * ξ[3, i]
    for r in 2:(2K + 1)
        coef[r, i] -= dt * p.μab * gcoef[r, i]
    end
    enforce_shape!(coef, i, K)
    xw, sx = wrap(xn, Lx)
    yw, sy = wrap(yn, Ly)
    x[i] = xw
    y[i] = yw
    imgx[i] += sx
    imgy[i] += sy
end

function integrate!(bodies::Bodies, g::Gradients, ξ, types::Tuple, dt, box::PeriodicBox, basis::FourierBasis)
    euler_maruyama_kernel!(get_backend(bodies.x), WORKGROUP)(bodies.x, bodies.y, bodies.θ,
        bodies.imgx, bodies.imgy, bodies.coef, bodies.type, g.gx, g.gy, g.gθ, g.gcoef, ξ,
        types, dt, box.Lx, box.Ly, nmodes(basis); ndrange = nbodies(bodies))
    return bodies
end
