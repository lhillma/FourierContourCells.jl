"""
    BodyType{T}(; v0 = 0, Dr = 0, Dt = 0, μr = 1, μθ = 1, μab = 0.1)

Dynamical parameters shared by all bodies of one type (e.g. cells, nuclei):

- `v0`:  self-propulsion speed along the orientation θ
- `Dr`:  rotational diffusion coefficient of θ (noise √(2Dr) ξ, persistence
  time 1/Dr)
- `Dt`:  translational diffusion coefficient (noise √(2Dt) ξ on x and y; 0 in
  the original model)
- `μr`, `μθ`, `μab`: mobilities of position, orientation and shape coefficients

The model has no temperature: `Dr` and `Dt` are the only noise amplitudes and
the shape modes are noiseless. Values are in reduced units (length R₀, energy
k R₀² with k the overlap coupling, time 1/(μr k)); see the README section
"Units and noise". Defaults follow Saito & Ishihara (2024), doi:10.1126/sciadv.adi8433, except
`v0` and `Dr`.
"""
struct BodyType{T}
    v0::T
    Dr::T
    Dt::T
    μr::T
    μθ::T
    μab::T
end

BodyType{T}(; v0 = 0, Dr = 0, Dt = 0, μr = 1, μθ = 1, μab = 0.1) where {T} =
    BodyType{T}(v0, Dr, Dt, μr, μθ, μab)
BodyType(; kwargs...) = BodyType{Float64}(; kwargs...)
BodyType{T}(b::BodyType) where {T} = BodyType{T}(b.v0, b.Dr, b.Dt, b.μr, b.μθ, b.μab)

"""
    Bodies

Structure-of-arrays state of all bodies. Every field is an array on the
compute backend, so the same container is used on CPU and GPU.

- `x`, `y`: centre positions wrapped into the box
- `imgx`, `imgy`: periodic image counters (unwrapped x = x + imgx·Lx)
- `θ`: orientation; it sets both the shape frame and the propulsion direction
- `coef`: `ncoef × N` body-frame Fourier coefficients `[a₀; aₙ; bₙ]`
- `R0`: radius scale (area πR0² under the hard area constraint)
- `type`: body type index into the simulation's `BodyType` table
- `parent`: id of the owning body (a cell is its own parent; a nucleus has
  its cell as parent). Pair terms can distinguish related bodies.
"""
struct Bodies{V <: AbstractVector, Mat <: AbstractMatrix, VI <: AbstractVector{Int32}}
    x::V
    y::V
    θ::V
    imgx::VI
    imgy::VI
    coef::Mat
    R0::V
    type::VI
    parent::VI
end

"""
    Bodies(x, y, θ, coef, R0; type = 1, parent = 1:N)

Construct a host-side body state. `type` may be a scalar or a vector.
"""
function Bodies(x::AbstractVector{T}, y, θ, coef, R0; type = 1, parent = 1:length(x)) where {T}
    N = length(x)
    all(length.((y, θ, R0)) .== N) && size(coef, 2) == N ||
        throw(DimensionMismatch("inconsistent body array lengths"))
    typ = type isa Integer ? fill(Int32(type), N) : Int32.(collect(type))
    return Bodies(collect(T, x), collect(T, y), collect(T, θ), zeros(Int32, N),
        zeros(Int32, N), Matrix{T}(coef), collect(T, R0), typ, Int32.(collect(parent)))
end

nbodies(b::Bodies) = length(b.x)

to_backend(backend, b::Bodies) = Bodies(map(f -> to_backend(backend, getfield(b, f)), fieldnames(Bodies))...)
to_host(b::Bodies) = Bodies(map(f -> to_host(getfield(b, f)), fieldnames(Bodies))...)

Base.show(io::IO, b::Bodies) = print(io, "Bodies(N=$(nbodies(b)), ncoef=$(size(b.coef, 1)))")
