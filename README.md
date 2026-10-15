# FourierContourCells.jl

A Julia implementation of the **Fourier contour cell model** for deformable,
self-propelled cells, generalised to typed bodies (cells, nuclei, …) and
user-defined energy terms. The compute kernels are written with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), so
the same code runs multithreaded on the CPU and, in a later step, on the GPU.

The model was introduced in

> N. Saito and S. Ishihara, *Cell deformability drives fluid-to-fluid phase
> transition in active cell monolayers*, Sci. Adv. **10**, eadi8433 (2024).
> Reference code: [Zenodo 10356125](https://zenodo.org/records/10356125).

and used, with polydisperse radii, in

> S. Pal, N. Saito, T. Kawasaki and H. Ebata, *Dense HeLa cell monolayers remain
> liquid-like despite strong crowding*, arXiv:2609.18768 (2026).

## Quick start

The repository is a Julia workspace: the package lives at the root, `test/` and
`examples/` are separate environments sharing one Manifest. Inside the Nix shell
(`nix develop`):

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia -t auto --project=examples examples/live.jl      # interactive window
julia -t auto --project=examples examples/movie.jl     # writes output/soft_fluid.mp4
julia -t auto --project=test test/runtests.jl          # test suite
```

```julia
using FourierContourCells, GLMakie

bodies, box = hexagonal_monolayer(32, 32; packing = 0.95)   # N = 1024, Lx:Ly = 2:√3
sim = Simulation(bodies, box;
    types = (BodyType(v0 = 0.06, Dr = 0.01, μab = 0.1),),   # per-type dynamics
    energies = (Tension(0.01), Overlap(1.0)),                # η = 0.01, excluded volume
    dt = 0.005)

step!(sim, 10_000)
shape_index(sim)                       # perimeter / √area per cell
live(sim; steps_per_frame = 50)        # watch it run (run/pause, speed slider)
record_movie(sim, "run.mp4"; frames = 300, steps_per_frame = 100)
```

## Model

Each body i has a centre **r**ᵢ, an orientation θᵢ (propulsion direction and
shape frame) and Fourier coefficients of its contour

  Rᵢ(φ) = R₀,ᵢ [a₀ + Σₙ₌₂ᴹ aₙ cos n(φ−θᵢ) + bₙ sin n(φ−θᵢ)],

with a₀ = √(1 − Σₙ(aₙ² + bₙ²)/2) fixing the area to πR₀² and the shape bound
Σₙ √(aₙ² + bₙ²) ≤ √(2/3) keeping R > 0. The overdamped dynamics

  ṙ = v₀ e(θ) − μr ∂E/∂r,  θ̇ = −μθ ∂E/∂θ + √(2Dr) ξ,  ȧₙ = −μab ∂E/∂aₙ

is integrated with Euler–Maruyama; contour integrals use `nθ` = 40 nodes.

## Energy terms

Energies act on bodies through their contour samples R(φₖ), R′(φₖ), R″(φₖ).
The library maps contour gradients to (**r**, θ, aₙ, bₙ) by the chain rule
(dense matrix products), so new terms never deal with Fourier coefficients.

**Single-body terms** (`BodyTerm`) have the form E = F(Q₁,…,Qₘ) with
Qⱼ = ∮ qⱼ(R, R′, R″) dφ. Define the densities and F; derivatives come from
automatic differentiation (ForwardDiff by default, Enzyme via
`ad_backend(::MyTerm) = EnzymeAD()` after `using Enzyme`) unless you provide
them by hand:

```julia
struct PerimeterElasticity{T} <: BodyTerm   # built in; shown as a template
    λ::T
    p0::T
end
densities(::PerimeterElasticity, R, dR, ddR) = sqrt(R^2 + dR^2)
combine(t::PerimeterElasticity, Q, body) = t.λ * (Q[1] - t.p0 * sqrt(π) * body.R0)^2
# optional, for speed: density_gradients(...) and combine_gradient(...)
```

**Contact terms** (`ContactTerm`) give the energy density g(d) felt by a contour
point at penetration depth d into a neighbour (d > 0 inside). The energy of
body i is Eᵢ = Σⱼ ∫_{Ωᵢ} g(dⱼ) dA, whose shape derivative is g on the contour,
so g itself is all that is needed:

```julia
pressure(o::Overlap, d, tᵢ, tⱼ, related) = d ≥ 0 ? J[tᵢ, tⱼ] : 0   # (schematic)
```

Built-in terms: `Tension(η)` (paper), `PerimeterElasticity(λ, p0)`,
`Overlap(J; related)` (paper's excluded volume, with CPM-like coupling tables
per type pair and for bodies with the same parent), `LinearRepulsion(k)`.
`OnTypes(term, types...)` restricts any term to body types. See
`examples/custom_energy.jl` for bending and adhesion terms.

**Nuclei.** Bodies carry a `type` and a `parent`. A nucleus is a body whose
parent is its cell; a negative `related` coupling in `Overlap` rewards overlap
with the own cell (equivalently: penalises nucleus area outside it) while
positive couplings repel all other cells and nuclei.

## Validation

- `test/data/reference_N64.txt` is output of the original C++ code (N = 64,
  Dr = 0). Positions, orientations and shape modes agree to ~1e-15 after
  4001 steps.
- Gradients are checked against finite differences for body terms (manual and
  AD) and contact terms, the cell list against brute force, and the overlap
  force of two discs against the exact result.

Deviations from the reference code, both deliberate:

1. The shape bound rescales **all** modes, as described in the paper; the
   reference `setR` only rescales n = 2 due to a loop bug. This only matters
   for strongly deformed cells hitting the bound.
2. Single-body terms exert no torque. They are rotation invariant, so their
   discrete θ-gradient is pure quadrature aliasing (~1e-8); the reference code
   omits it too.

## Performance

N = 1024 at packing 0.95, Float64: 4.7 ms/step on one thread (reference C++
12.0 ms/step), 1.6 ms/step with 8 threads.

## Layout

```
src/
  basis.jl          synthesis matrices, coefficient layout [a₀; aₙ; bₙ]
  bodies.jl         body state (structure of arrays) and BodyType
  geometry.jl       lab-frame rotation, contour synthesis, R(ψ) off-grid
  energies/         interface (+ AD), body terms, contact terms & kernel
  gradients.jl      contour gradients → (r, θ, aₙ, bₙ)
  neighbors.jl      cell list and per-body neighbour list
  dynamics.jl       Euler–Maruyama integrator, shape constraints
  simulation.jl     Simulation, step!, run!
ext/                Makie (visualisation) and Enzyme extensions
```
