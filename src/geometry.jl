"""
    Geometry

Quantities derived from the body state that energy terms and kernels read:

- `lab`: `ncoef × N` lab-frame coefficients `[a₀; αₙ; βₙ]`
- `R`, `dR`, `ddR`: `nθ × N` contour radius and its first two φ-derivatives
  at the quadrature nodes
- `rmax`: upper bound of R(φ) per body, R₀(a₀ + Σₙ √(aₙ² + bₙ²))
"""
struct Geometry{V <: AbstractVector, Mat <: AbstractMatrix}
    lab::Mat
    R::Mat
    dR::Mat
    ddR::Mat
    rmax::V
end

function Geometry(basis::FourierBasis, bodies::Bodies)
    N = nbodies(bodies)
    T = eltype(bodies.x)
    backend = get_backend(bodies.x)
    alloc(dims...) = KernelAbstractions.zeros(backend, T, dims...)
    return Geometry(alloc(ncoef(basis), N), alloc(basis.nθ, N), alloc(basis.nθ, N),
        alloc(basis.nθ, N), alloc(N))
end

@kernel function lab_frame_kernel!(lab, rmax, @Const(coef), @Const(θ), @Const(R0), K)
    i = @index(Global)
    lab[1, i] = coef[1, i]
    bound = coef[1, i]
    for m in 1:K
        n = m + 1
        a = coef[1 + m, i]
        b = coef[1 + K + m, i]
        s, c = sincos(n * θ[i])
        lab[1 + m, i] = a * c - b * s
        lab[1 + K + m, i] = a * s + b * c
        bound += sqrt(a^2 + b^2)
    end
    rmax[i] = R0[i] * bound
end

"""
    update_geometry!(geom, basis, bodies)

Rotate coefficients into the lab frame and synthesise the contours of all
bodies at the quadrature nodes (three dense matrix products).
"""
function update_geometry!(geom::Geometry, basis::FourierBasis, bodies::Bodies)
    backend = get_backend(bodies.x)
    lab_frame_kernel!(backend, WORKGROUP)(geom.lab, geom.rmax, bodies.coef, bodies.θ,
        bodies.R0, nmodes(basis); ndrange = nbodies(bodies))
    R0row = reshape(bodies.R0, 1, :)
    mul!(geom.R, basis.C, geom.lab)
    geom.R .*= R0row
    mul!(geom.dR, basis.D, geom.lab)
    geom.dR .*= R0row
    mul!(geom.ddR, basis.DD, geom.lab)
    geom.ddR .*= R0row
    return geom
end

"""
    contour_radius(lab, j, R0, K, ψ)

Evaluate R_j(ψ) of body `j` at an arbitrary lab-frame angle `ψ` from its
lab-frame coefficients. Uses the Chebyshev recurrence for cos nψ, sin nψ.
"""
@inline function contour_radius(lab, j, R0, K, ψ)
    s1, c1 = sincos(ψ)
    # (c, s) holds (cos nψ, sin nψ), start at n = 1
    c, s = c1, s1
    r = lab[1, j]
    for m in 1:K
        c, s = c * c1 - s * s1, s * c1 + c * s1
        r += lab[1 + m, j] * c + lab[1 + K + m, j] * s
    end
    return R0 * r
end
