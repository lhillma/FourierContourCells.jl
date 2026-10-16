# FourierContourCells.jl

A Julia implementation of the **Fourier contour cell model** for deformable,
self-propelled cells, generalised to typed bodies (cells, nuclei, …) and
user-defined energy terms. The compute kernels are written with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), so
the same code runs multithreaded on the CPU and, in a later step, on the GPU.

This package is an **independent reimplementation**. The model, its equations
of motion and the numerical scheme are entirely due to Nen Saito and Shuji
Ishihara; it is not affiliated with or endorsed by them. Please cite their work
(see [Citing](#citing)) when you use this code.

> N. Saito and S. Ishihara, *Cell deformability drives fluid-to-fluid phase
> transition in active cell monolayers*, Sci. Adv. **10**, eadi8433 (2024).
> DOI: [10.1126/sciadv.adi8433](https://doi.org/10.1126/sciadv.adi8433)
>
> Reference code and data: N. Saito and S. Ishihara, Zenodo (2023), CC BY 4.0.
> DOI: [10.5281/zenodo.10356125](https://doi.org/10.5281/zenodo.10356125)

Polydisperse radii and the reduced units follow

> S. Pal, N. Saito, T. Kawasaki and H. Ebata, *Dense HeLa cell monolayers remain
> liquid-like despite strong crowding*, arXiv:2609.18768 (2026).
> DOI: [10.48550/arXiv.2609.18768](https://doi.org/10.48550/arXiv.2609.18768)

## Quick start

The repository is a Julia workspace: the package lives at the root, `test/` and
`examples/` are separate environments sharing one Manifest. Inside the Nix shell
(`nix develop`):

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia -t auto --project=examples examples/live.jl      # interactive window
julia -t auto --project=examples examples/movie.jl     # writes output/soft_fluid.mp4
julia -t auto --project=examples examples/nucleus.jl   # cells with nuclei (live window)
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
FourierContourCells.live_view(sim; steps_per_frame = 50)   # run/pause, speed slider
FourierContourCells.record_movie(sim, "run.mp4"; frames = 300, steps_per_frame = 100)
```

## Model

Each body i has a centre **r**ᵢ, an orientation θᵢ (propulsion direction and
shape frame) and Fourier coefficients of its contour

  Rᵢ(φ) = R₀,ᵢ [a₀ + Σₙ₌₂ᴹ aₙ cos n(φ−θᵢ) + bₙ sin n(φ−θᵢ)],

with a₀ = √(1 − Σₙ(aₙ² + bₙ²)/2) fixing the area to πR₀² and the shape bound
Σₙ √(aₙ² + bₙ²) ≤ √(2/3) keeping R > 0. The overdamped dynamics

  ṙ = v₀ e(θ) − μr ∂E/∂r,  θ̇ = −μθ ∂E/∂θ + √(2Dr) ξ,  ȧₙ = −μab ∂E/∂aₙ

is integrated with Euler–Maruyama; contour integrals use `nθ` = 40 nodes.

### Units and noise

The model is **athermal and active**. It contains the following stochastic
forces:

- rotational noise on the polarity θ with diffusion coefficient `Dr`
  (persistence time 1/Dr), which together with `v0` makes cells active
  Brownian particles, and
- optional translational noise with diffusion coefficient `Dt` (0 in the
  paper, so positions only move by self-propulsion and forces).

All quantities are in reduced units, following Saito & Ishihara and Pal et al.:

| scale  | unit                    | set by                                         |
|:-------|:------------------------|:-----------------------------------------------|
| length | R₀ (cell radius)        | `R0 = 1` in `hexagonal_monolayer`              |
| energy | k R₀²                   | overlap energy per unit area k: `Overlap(1.0)` |
| time   | t₀ = 1/(μr k)           | translational mobility `μr = 1`                |

In these units the parameters are dimensionless combinations: tension
η/(kR₀), speed v₀/(μr k R₀), rotational diffusion Dr/(μr k), translational
diffusion Dt/(μr k R₀²), and mobilities μθR₀²/μr and μabR₀²/μr for the
orientation and the (dimensionless) shape modes. Pal et al. use the cell
diameter L₀ = 2R₀ as length unit instead; their η and v₀ differ from ours by
factors of 2 accordingly.

## Energy terms

Energies act on bodies through their contour samples R(φₖ), R′(φₖ), R″(φₖ).
The library maps contour gradients to (**r**, θ, aₙ, bₙ) by the chain rule
(dense matrix products), so new terms never deal with Fourier coefficients.

**Single-body terms** (`BodyTerm`) are built from contour integrals in two
stages: `contour_integrands(term, R, R′, R″)` returns integrands qⱼ, which the
library integrates to Qⱼ = ∮ qⱼ dφ, and `energy(term, Q, body)` returns
E = F(Q) (default: `sum(Q)`, i.e. the integrand is an energy density).
Derivatives come from automatic differentiation (ForwardDiff by default,
Enzyme via `ad_backend(::MyTerm) = EnzymeAD()` after `using Enzyme`) unless
you provide them by hand:

```julia
import FourierContourCells: contour_integrands, energy

struct Bending{T} <: BodyTerm      # E = κ ∮ c² ds: integrand only
    κ::T
end
contour_integrands(t::Bending, R, dR, ddR) = t.κ * (R^2 + 2dR^2 - R * ddR)^2 / (R^2 + dR^2)^(5 / 2)

struct PerimeterElasticity{T} <: BodyTerm   # E = λ (l - l₀)²: nonlinear in Q
    λ::T
    p0::T
end
contour_integrands(::PerimeterElasticity, R, dR, ddR) = sqrt(R^2 + dR^2)   # Q[1] = perimeter
energy(t::PerimeterElasticity, Q, body) = t.λ * (Q[1] - t.p0 * sqrt(π) * body.R0)^2
```

`examples/custom_energy.jl` is a step-by-step tutorial, including hand-written
derivatives and a contact term.

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
`OnTypes(term, types...)` restricts any term to body types.

**Bond terms** (`BondTerm`) are pair energies E(**r**ᵢ − **r**ⱼ) between the
centres of explicitly listed bodies, `Simulation(...; bonds = [(i, j), …])`.
Define `bond_energy(term, Δ, tᵢ, tⱼ)` (symmetric under Δ → −Δ with swapped
types); the gradient comes from AD unless `bond_gradient` is given.
Built-in: `HarmonicBond(k; r0)`.

**Nuclei.** Bodies carry a `type` and a `parent`. A nucleus is a body whose
parent is its cell (`with_nuclei(cells; ratio)` adds one per cell):

```julia
bodies = with_nuclei(cells; ratio = 0.5)          # types 1 (cells) and 2 (nuclei)
sim = Simulation(bodies, box;
    types = (BodyType(v0 = 0.06, Dr = 0.01),               # cell
             BodyType(v0 = 0.0, Dr = 0.01, drag = :parent)),  # nucleus
    energies = (OnTypes(Tension(0.01), 1), OnTypes(Tension(0.05), 2),
                Overlap([1.0 1.0; 1.0 1.0]; related = [0 -2.0; -2.0 0]),
                HarmonicBond(0.2)),
    bonds = parent_bonds(bodies))
FourierContourCells.live_view(sim; layers = ((types = 1, color = :shape_index),
                                             (types = 2, color = :steelblue)))
```

- `drag = :parent`: the nucleus feels drag against its cell's interior
  instead of the substrate. The cell bears the nucleus's forces and carries
  it along, so an isolated cell keeps its speed v0, and parent–child bonds
  are internal forces.
- A negative `related` coupling makes cell and own nucleus gain energy from
  overlapping. The membrane wraps the nucleus and pulls back parts that poke
  out; a nucleus fully inside its cell feels no force from it.
- `HarmonicBond` on `parent_bonds` anchors each nucleus to its cell centre
  (a crude cytoskeleton).
- Diagnostics: `protrusion(sim)` and the `type =` filter of `shape_index`,
  `areas`, `perimeters`.

`examples/nucleus.jl` documents the modelling choices and checks them.
For quick experiments, `Simulation(...; hook! = f)` calls `f(sim)` after the
gradients in every step. It is meant for prototyping only.

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
  bodies.jl         body state (structure of arrays), BodyType, parent → children table
  geometry.jl       lab-frame rotation, contour synthesis, R(ψ) off-grid
  energies/         interface (+ AD), body, contact and bond terms & kernels
  gradients.jl      contour gradients → (r, θ, aₙ, bₙ)
  neighbors.jl      cell list and per-body neighbour list
  dynamics.jl       displacements (substrate / parent drag), Euler–Maruyama, shape constraints
  simulation.jl     Simulation, step!, run!
ext/                Makie (visualisation) and Enzyme extensions
```

## Citing

If you use this package, please cite the original model paper and, where
relevant, the reference implementation (also listed in `CITATION.cff`, which
GitHub shows as "Cite this repository"):

```bibtex
@article{saito2024cell,
  author  = {Saito, Nen and Ishihara, Shuji},
  title   = {Cell deformability drives fluid-to-fluid phase transition in active cell monolayers},
  journal = {Science Advances},
  volume  = {10},
  number  = {19},
  pages   = {eadi8433},
  year    = {2024},
  doi     = {10.1126/sciadv.adi8433}
}

@software{saito2023code,
  author    = {Saito, Nen and Ishihara, Shuji},
  title     = {Cell Deformability Drives Fluid-to-Fluid Phase Transition in Active Cell Monolayers (code and data)},
  publisher = {Zenodo},
  year      = {2023},
  doi       = {10.5281/zenodo.10356125}
}
```

`test/data/reference_N64.txt` contains output of the authors' reference code
(CC BY 4.0, [doi:10.5281/zenodo.10356125](https://doi.org/10.5281/zenodo.10356125)),
run with the modifications listed in its header. The reference code itself is
not redistributed here.

