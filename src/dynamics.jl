# Equations of motion of the Fourier contour cell model, Eq. (5) and Materials
# and Methods of N. Saito and S. Ishihara, Sci. Adv. 10, eadi8433 (2024), doi:10.1126/sciadv.adi8433;
# the update order follows the authors' reference code (Zenodo,
# doi:10.5281/zenodo.10356125).
#
# Overdamped Langevin dynamics, integrated with the Euler–Maruyama scheme:
#
#   ṙ  = v0 e(θ) - μr ∂E/∂r + √(2Dt) ξ
#   θ̇  = -μθ ∂E/∂θ + √(2Dr) ξ
#   ȧₙ = -μab ∂E/∂aₙ,  ḃₙ = -μab ∂E/∂bₙ
#
# with independent standard white noises ξ. Bodies whose type has
# drag = :parent (e.g. nuclei) feel drag relative to their parent instead and
# are carried along with it (see displacement_kernel!). The dynamics is active and
# athermal: Dr and Dt are free noise amplitudes, not tied to a temperature,
# and the shape modes carry no noise. Reduced units: length R₀, energy k R₀²
# (k = overlap coupling), time 1/(μr k).

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

Constraint and bound as in Saito & Ishihara (2024), doi:10.1126/sciadv.adi8433
(Materials and Methods). Note: the reference implementation
(FourierContourCell.cpp, `setR`, doi:10.5281/zenodo.10356125) only rescales
the n = 2 mode because of a loop bug; here all modes are rescaled as described
in the paper.
"""
Base.@propagate_inbounds function enforce_shape!(coef, i, K)
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
    @inbounds begin
        i = @index(Global)
        enforce_shape!(coef, i, K)
    end
end

function enforce_shape!(bodies::Bodies, basis::FourierBasis)
    enforce_shape_kernel!(get_backend(bodies.coef), WORKGROUP)(bodies.coef, nmodes(basis);
        ndrange = nbodies(bodies))
    return bodies
end

# Translational displacement of a body with substrate drag, including the
# forces on its children with drag = :parent, which it bears:
#   Δr = dt (v0 e(θ) - μr (∂E/∂r + Σ_children ∂E/∂r_child)) + √(2 Dt dt) (ξ₁, ξ₂)
Base.@propagate_inbounds function substrate_displacement(i, θ, type, gx, gy, ξ, cidx, ccount, types, dt)
    p = types[type[i]]
    fx = gx[i]
    fy = gy[i]
    for q in 1:ccount[i]
        k = cidx[q, i]
        if types[type[k]].parent_drag
            fx += gx[k]
            fy += gy[k]
        end
    end
    s, c = sincos(θ[i])
    σt = sqrt(2 * p.Dt * dt)
    return dt * (p.v0 * c - p.μr * fx) + σt * ξ[1, i], dt * (p.v0 * s - p.μr * fy) + σt * ξ[2, i]
end

# Translational displacement of every body in one Euler–Maruyama step. One
# work item per body i, with p = types[type[i]].
#
# Substrate drag (p.parent_drag false): see `substrate_displacement`.
# Parent drag: body i with parent c feels drag ζ_i = 1/μr_i against the
# interior of c, and c feels the reaction. The force balances
#   ζ_c ṙ_c - Σ_children ζ_k (ṙ_k - ṙ_c) = F_c,   ζ_k (ṙ_k - ṙ_c) = F_k
# (forces F = -∂E/∂r, plus self-propulsion v0 e(θ) as a velocity) are solved
# explicitly by
#   ṙ_c = v0_c e(θ_c) + μr_c (F_c + Σ_children F_k),   ṙ_k = ṙ_c + v0_k e(θ_k) + μr_k F_k,
# so internal forces between parent and child (e.g. a bond) cancel in the
# parent's motion. A child repeats its parent's computation (same ξ, θ, ∂E/∂r)
# and adds its own relative displacement and noise.
#
#   dx, dy       (out) N, displacements
#   θ, type      N, orientations (before the update) and body types
#   parent       N, parent ids
#   gx, gy       N, ∂E/∂x, ∂E/∂y
#   ξ            3 × N standard normal random numbers (rows 1, 2 used)
#   cidx, ccount children of each body (see ChildList)
#   types        tuple of BodyType;  dt: time step
#   status       run status; nothing is done while it is frozen (see isactive)
@kernel function displacement_kernel!(dx, dy, @Const(θ), @Const(type), @Const(parent), @Const(gx),
        @Const(gy), @Const(ξ), @Const(cidx), @Const(ccount), types, dt, status)
    @inbounds if isactive(status)
        i = @index(Global)
        p = types[type[i]]
        if p.parent_drag
            c = parent[i]
            ux, uy = substrate_displacement(c, θ, type, gx, gy, ξ, cidx, ccount, types, dt)
            s, co = sincos(θ[i])
            σt = sqrt(2 * p.Dt * dt)
            dx[i] = ux + dt * (p.v0 * co - p.μr * gx[i]) + σt * ξ[1, i]
            dy[i] = uy + dt * (p.v0 * s - p.μr * gy[i]) + σt * ξ[2, i]
        else
            dx[i], dy[i] = substrate_displacement(i, θ, type, gx, gy, ξ, cidx, ccount, types, dt)
        end
    end
end

# One Euler–Maruyama step for every body. One work item per body i.
#
# With p = types[type[i]] the parameters of body i's type and ξ standard normal
# numbers, the update reads
#   r  ← r + Δr   (translational displacement from displacement_kernel!)
#   θ  ← θ - dt μθ ∂E/∂θ + √(2 Dr dt) ξ₃
#   aₙ ← aₙ - dt μab ∂E/∂aₙ,  bₙ ← bₙ - dt μab ∂E/∂bₙ
# Afterwards the shape constraints are enforced and the position is wrapped
# into the box, counting crossed periods in the image counters.
#
#   x, y         (in/out) N, positions
#   θ            (in/out) N, orientations
#   imgx, imgy   (in/out) N, periodic image counters
#   coef         (in/out) ncoef × N body-frame coefficients
#   type         N, body types (index into `types`)
#   dx, dy       N, translational displacements
#   gθ           N, energy gradient ∂E/∂θ
#   gcoef        ncoef × N, ∂E/∂[a₀; aₙ; bₙ] (row 1 unused)
#   ξ            3 × N standard normal random numbers (row 3 used)
#   types        tuple of BodyType (isbits, indexed with the type)
#   dt           time step; Lx, Ly: box size; K: number of shape modes
#   clock        (in/out) device step counter, incremented once (or nothing)
#   status       run status; the kernel does nothing while it is frozen (see isactive)
@kernel function euler_maruyama_kernel!(x, y, θ, imgx, imgy, coef, @Const(type), @Const(dx),
        @Const(dy), @Const(gθ), @Const(gcoef), @Const(ξ), types, dt, Lx, Ly, K, clock, status)
    @inbounds if isactive(status)
        i = @index(Global)
        tick!(clock, i)
        p = types[type[i]]
        xn = x[i] + dx[i]
        yn = y[i] + dy[i]
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
end

"""
    integrate!(bodies, grad, disp, ξ, children, types, dt, box, basis; clock, status)

One Euler–Maruyama step: translational displacements into `disp` (2 × N),
then the update of all body variables. `clock` (device step counter) is
incremented; nothing is updated while `status` is frozen.
"""
function integrate!(bodies::Bodies, g::Gradients, disp, ξ, children::ChildList, types::Tuple, dt,
        box::PeriodicBox, basis::FourierBasis; clock = nothing, status = nothing)
    backend = get_backend(bodies.x)
    N = nbodies(bodies)
    dx = view(disp, 1, :)
    dy = view(disp, 2, :)
    displacement_kernel!(backend, WORKGROUP)(dx, dy, bodies.θ, bodies.type, bodies.parent, g.gx,
        g.gy, ξ, children.idx, children.count, types, dt, status; ndrange = N)
    euler_maruyama_kernel!(backend, WORKGROUP)(bodies.x, bodies.y, bodies.θ, bodies.imgx,
        bodies.imgy, bodies.coef, bodies.type, dx, dy, g.gθ, g.gcoef, ξ, types, dt, box.Lx,
        box.Ly, nmodes(basis), clock, status; ndrange = N)
    return bodies
end
