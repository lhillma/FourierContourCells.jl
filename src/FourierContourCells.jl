"""
    FourierContourCells

Independent Julia implementation of the Fourier contour cell model,
generalised to typed bodies (cells, nuclei, ...) and user-defined energy terms.
Kernels are written with KernelAbstractions.jl so the same code runs on CPU and
GPU.

The model, its equations of motion and its numerical scheme are due to

    N. Saito and S. Ishihara, "Cell deformability drives fluid-to-fluid phase
    transition in active cell monolayers", Sci. Adv. 10, eadi8433 (2024),
    doi:10.1126/sciadv.adi8433

with reference code by the same authors (CC BY 4.0):

    N. Saito and S. Ishihara, Zenodo (2023), doi:10.5281/zenodo.10356125

Polydisperse radii and the reduced-unit conventions follow

    S. Pal, N. Saito, T. Kawasaki and H. Ebata, "Dense HeLa cell monolayers
    remain liquid-like despite strong crowding", arXiv:2609.18768 (2026),
    doi:10.48550/arXiv.2609.18768

Please cite these works when using this package (see README and CITATION.cff).
"""
module FourierContourCells

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend, CPU
using LinearAlgebra: mul!
using Random: Random
using StaticArrays: SVector, SMatrix
import ForwardDiff

include("utils.jl")
include("random.jl")
include("basis.jl")
include("box.jl")
include("bodies.jl")
include("geometry.jl")
include("energies/interface.jl")
include("energies/body_terms.jl")
include("gradients.jl")
include("neighbors.jl")
include("energies/contact.jl")
include("energies/bonds.jl")
include("dynamics.jl")
include("simulation.jl")
include("initial.jl")
include("observables.jl")
include("visualization.jl")

# geometry and state
export FourierBasis, PeriodicBox, Bodies, BodyType, nbodies, ncoef, nmodes
# energy interface: types are exported; the methods users extend are public but
# not exported (generic names), e.g. `FourierContourCells.energy(t::MyTerm, Q, b) = ...`
export EnergyTerm, BodyTerm, ContactTerm, BondTerm, BodyInfo, OnTypes, ForwardDiffAD, EnzymeAD
public contour_integrands, energy, contour_integrand_gradients, energy_gradient, pressure,
    interaction_range, applies_to, ad_backend, bond_energy, bond_gradient
# built-in energy terms
export Tension, PerimeterElasticity, Overlap, LinearRepulsion, HarmonicBond
# simulation
export Simulation, step!, run!, compute_gradients!, hexagonal_monolayer, with_nuclei, parent_bonds, CPU,
    KernelLayout, BodyParallel, NodeParallel
# observables
export perimeters, areas, shape_index, positions, unwrapped_positions, contours, body_energy,
    protrusion
# helpers for user kernels and diagnostics (public, not exported)
public minimum_image, contour_radius, node_angle, to_host, to_backend, default_layout
# visualisation (Makie extension)
export cellview!, update_view!, cellplot
# public API with generic names, not exported: call them qualified,
# e.g. `FourierContourCells.live_view(sim)`
public live_view, record_movie

end
