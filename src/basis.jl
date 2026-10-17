"""
    FourierBasis{T}(; maxmode = 6, nθ = 40, backend = CPU())

Quadrature nodes and dense synthesis matrices for contours

    R(φ) = R₀ [a₀ + Σₙ (αₙ cos nφ + βₙ sin nφ)],   n = 2, …, maxmode,

sampled at the `nθ` equispaced lab-frame angles φₖ = (k-1)·2π/nθ.

Coefficients of one body are stored as a column of length [`ncoef`](@ref)
in the order `[a₀; a₂ … a_M; b₂ … b_M]`. The lab-frame coefficients (αₙ, βₙ)
are the body-frame coefficients (aₙ, bₙ) rotated by the body orientation θ,
so the matrices are shared by all bodies and contour synthesis for all bodies
is one matrix product `C * lab` (see [`update_geometry!`](@ref)).

Fields:
- `C`, `D`, `DD`: `nθ × ncoef` matrices mapping lab coefficients to
  R̂, dR̂/dφ, d²R̂/dφ² (R = R₀ R̂).
- `S`: `3nθ × ncoef`, the stacked `[C; D; DD]`, which synthesises R, R', R''
  of all bodies in a single matrix product.
- `Δφ`: quadrature weight 2π/nθ.
"""
struct FourierBasis{T, Mat <: AbstractMatrix{T}}
    nθ::Int
    maxmode::Int
    Δφ::T
    C::Mat
    D::Mat
    DD::Mat
    S::Mat
end

function FourierBasis{T}(; maxmode::Integer = 6, nθ::Integer = 40, backend = CPU()) where {T}
    maxmode >= 2 || throw(ArgumentError("maxmode must be ≥ 2"))
    # Sampling theorem: the highest mode must be resolved by the nodes
    nθ > 2maxmode || throw(ArgumentError("nθ must exceed 2·maxmode"))
    K = maxmode - 1
    nc = 2K + 1
    Δφ = 2π / nθ
    C = zeros(T, nθ, nc)
    D = zeros(T, nθ, nc)
    DD = zeros(T, nθ, nc)
    for k in 1:nθ
        φ = (k - 1) * Δφ
        C[k, 1] = 1
        for (m, n) in enumerate(2:maxmode)
            s, c = sincos(n * φ)
            C[k, 1 + m] = c
            C[k, 1 + K + m] = s
            D[k, 1 + m] = -n * s
            D[k, 1 + K + m] = n * c
            DD[k, 1 + m] = -n^2 * c
            DD[k, 1 + K + m] = -n^2 * s
        end
    end
    return FourierBasis(nθ, maxmode, T(Δφ), to_backend(backend, C),
        to_backend(backend, D), to_backend(backend, DD), to_backend(backend, vcat(C, D, DD)))
end
FourierBasis(; kwargs...) = FourierBasis{Float64}(; kwargs...)

"Number of shape modes per sine/cosine family (modes 2, …, maxmode)."
nmodes(b::FourierBasis) = b.maxmode - 1

"Length of a body's coefficient column `[a₀; aₙ; bₙ]`."
ncoef(b::FourierBasis) = 2nmodes(b) + 1

"Lab-frame angle of contour node `k` (1-based)."
node_angle(b::FourierBasis{T}, k) where {T} = (k - 1) * b.Δφ

Base.show(io::IO, b::FourierBasis{T}) where {T} =
    print(io, "FourierBasis{$T}(maxmode=$(b.maxmode), nθ=$(b.nθ))")
