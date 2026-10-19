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
    ξ1, ξ2 = noise_xy(ξ, typeof(dt), i)
    return dt * (p.v0 * c - p.μr * fx) + σt * ξ1, dt * (p.v0 * s - p.μr * fy) + σt * ξ2
end

# Translational displacement and orientation noise of every body in one
# Euler–Maruyama step. One work item per body i, with p = types[type[i]].
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
# and adds its own relative displacement and noise. Row 3 gets the
# orientation noise √(2 Dr dt) ξ₃, so that the update kernel needs no noise.
# (Displacement and update stay separate kernels: a child reads its parent's
# θ before the update.)
#
#   disp         (out) 3 × N, (Δx, Δy, √(2 Dr dt) ξ₃) per body
#   θ, type      N, orientations (before the update) and body types
#   parent       N, parent ids
#   gx, gy       N, ∂E/∂x, ∂E/∂y
#   ξ            noise: a 3 × N matrix of standard normals, or PhiloxNoise
#   cidx, ccount children of each body (see ChildList)
#   types        tuple of BodyType;  dt: time step
#   status       run status; nothing is done while it is frozen (see isactive).
#                Work item 1 copies the rebuild flag to STATUS_UPDATE for the
#                update kernel.
@kernel function displacement_kernel!(disp, @Const(θ), @Const(type), @Const(parent), @Const(gx),
        @Const(gy), ξ, @Const(cidx), @Const(ccount), types, dt, status)
    @inbounds mirror_status!(status, @index(Global))
    @inbounds if isactive(status)
        i = @index(Global)
        p = types[type[i]]
        if p.parent_drag
            c = parent[i]
            ux, uy = substrate_displacement(c, θ, type, gx, gy, ξ, cidx, ccount, types, dt)
            s, co = sincos(θ[i])
            σt = sqrt(2 * p.Dt * dt)
            ξ1, ξ2 = noise_xy(ξ, typeof(dt), i)
            disp[1, i] = ux + dt * (p.v0 * co - p.μr * gx[i]) + σt * ξ1
            disp[2, i] = uy + dt * (p.v0 * s - p.μr * gy[i]) + σt * ξ2
        else
            disp[1, i], disp[2, i] = substrate_displacement(i, θ, type, gx, gy, ξ, cidx, ccount, types, dt)
        end
        disp[3, i] = sqrt(2 * p.Dr * dt) * noise_θ(ξ, typeof(dt), i)
    end
end

# Euler–Maruyama update of every body, fused with the per-body work around it.
# One work item per body i, with p = types[type[i]]:
#
# 1. If `frame`: the body-frame gradient gcoef from glab (see body_frame!; it
#    uses coef and θ before the update) and stores it in gcoef. Otherwise
#    gcoef must be current (e.g. from compute_gradients! before a hook).
# 2. The update
#      r  ← r + Δr   (translational displacement from displacement_kernel!)
#      θ  ← θ - dt μθ ∂E/∂θ + √(2 Dr dt) ξ₃
#      aₙ ← aₙ - dt μab ∂E/∂aₙ,  bₙ ← bₙ - dt μab ∂E/∂bₙ
#    then the shape constraints, and wrapping of the position into the box
#    (crossed periods are counted in the image counters).
# 3. If `lab` is given: lab-frame coefficients and contour bound of the new
#    state (see lab_frame!), so only the synthesis product remains.
# 4. If `skin` is given: the Verlet skin check of the new state (see
#    skin_check_kernel!), which raises the rebuild flag for the next step.
# 5. Work item 1 increments the device step clock.
#
#   x, y, θ      (in/out) N, positions and orientations
#   imgx, imgy   (in/out) N, periodic image counters
#   coef         (in/out) ncoef × N body-frame coefficients
#   gcoef        (in/out) ncoef × N, ∂E/∂[a₀; aₙ; bₙ] (out if frame)
#   lab, labR0, rmax  (out) lab-frame coefficients, R0·lab, contour bound (or nothing)
#   type, R0     N, body types and radius scales
#   disp         3 × N, from displacement_kernel!
#   gθ           N, ∂E/∂θ;  glab: ncoef × N, ∂E/∂lab / R0 from pullback! (if frame)
#   skin         nothing, or (x0, y0, rmax0, skin) of the neighbour list
#   types        tuple of BodyType;  dt: time step;  Lx, Ly: box;  K: shape modes
#   clock        (in/out) device step counter (or nothing)
#   status       (in/out) run status; nothing is done while STATUS_UPDATE is
#                set, the skin check may raise STATUS_REBUILD
@kernel function euler_maruyama_kernel!(x, y, θ, imgx, imgy, coef, gcoef, lab, labR0, rmax,
        @Const(type), @Const(R0), @Const(disp), @Const(gθ), @Const(glab), frame, skin, types, dt,
        Lx, Ly, K, clock, status)
    @inbounds if isactive(status, STATUS_UPDATE)
        i = @index(Global)
        p = types[type[i]]
        frame && body_frame!(gcoef, glab, coef, θ[i], R0[i], i, K)
        θn = θ[i] - dt * p.μθ * gθ[i] + disp[3, i]
        θ[i] = θn
        for r in 2:(2K + 1)
            coef[r, i] -= dt * p.μab * gcoef[r, i]
        end
        enforce_shape!(coef, i, K)
        xw, sx = wrap(x[i] + disp[1, i], Lx)
        yw, sy = wrap(y[i] + disp[2, i], Ly)
        x[i] = xw
        y[i] = yw
        imgx[i] += sx
        imgy[i] += sy
        lab === nothing || lab_frame!(lab, labR0, rmax, coef, θn, R0[i], i, K)
        skin === nothing || skin_flag!(status, skin, xw, yw, rmax[i], i, Lx, Ly)
        tick!(clock, i)
    end
end

"""
    integrate!(bodies, grad, disp, ξ, children, types, dt, box, basis;
               geom, skin, frame, clock, status)

One Euler–Maruyama step: displacements and orientation noise into `disp`
(3 × N), then the update of all body variables. `ξ` is a 3 × N matrix of
standard normals or a `PhiloxNoise`. Fused per-body work (see
`euler_maruyama_kernel!`): with `frame = true` the body-frame gradient is
computed from `grad.glab` (else `grad.gcoef` is used as is); with `geom`
the lab-frame coefficients of the new state are written to it (only the
synthesis product of `update_geometry!` remains); with `skin = (x0, y0,
rmax0, skin)` the Verlet skin check raises the rebuild flag in `status`.
`clock` (device step counter) is incremented; nothing is updated while
`status` is frozen.
"""
function integrate!(bodies::Bodies, g::Gradients, disp, ξ, children::ChildList, types::Tuple, dt,
        box::PeriodicBox, basis::FourierBasis; geom = nothing, skin = nothing, frame::Bool = false,
        clock = nothing, status = nothing)
    backend = get_backend(bodies.x)
    N = nbodies(bodies)
    displacement_kernel!(backend, WORKGROUP)(disp, bodies.θ, bodies.type, bodies.parent, g.gx,
        g.gy, ξ, children.idx, children.count, types, dt, status; ndrange = N)
    lab, labR0, rmax = geom === nothing ? (nothing, nothing, nothing) : (geom.lab, geom.labR0, geom.rmax)
    euler_maruyama_kernel!(backend, WORKGROUP)(bodies.x, bodies.y, bodies.θ, bodies.imgx,
        bodies.imgy, bodies.coef, g.gcoef, lab, labR0, rmax, bodies.type, bodies.R0, disp, g.gθ,
        g.glab, frame, skin, types, dt, box.Lx, box.Ly, nmodes(basis), clock, status; ndrange = N)
    return bodies
end
