"""
    FourierContourCells

Implementation of the Fourier contour cell model (N. Saito and S. Ishihara,
Sci. Adv. 10, eadi8433 (2024)), generalised to typed bodies (cells, nuclei, ...)
and user-defined energy terms. Kernels are written with KernelAbstractions.jl
so the same code runs on CPU and GPU.
"""
module FourierContourCells

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend, CPU
using LinearAlgebra: mul!
using Random: Random, randn!
using StaticArrays: SVector, SMatrix
import ForwardDiff

include("utils.jl")
include("basis.jl")
include("box.jl")
include("bodies.jl")
include("geometry.jl")
include("energies/interface.jl")
include("energies/body_terms.jl")
include("gradients.jl")
include("neighbors.jl")
include("energies/contact.jl")
include("dynamics.jl")
include("simulation.jl")
include("initial.jl")
include("observables.jl")
include("visualization.jl")

# geometry and state
export FourierBasis, PeriodicBox, Bodies, BodyType, nbodies, ncoef, nmodes
# energy interface: types are exported; the methods users extend are public but
# not exported (generic names), e.g. `FourierContourCells.energy(t::MyTerm, Q, b) = ...`
export EnergyTerm, BodyTerm, ContactTerm, BodyInfo, OnTypes, ForwardDiffAD, EnzymeAD
public contour_integrands, energy, contour_integrand_gradients, energy_gradient, pressure,
    interaction_range, applies_to, ad_backend
# built-in energy terms
export Tension, PerimeterElasticity, Overlap, LinearRepulsion
# simulation
export Simulation, step!, run!, compute_gradients!, hexagonal_monolayer, CPU
# observables
export perimeters, areas, shape_index, positions, unwrapped_positions, contours, body_energy
# visualisation (Makie extension)
export cellview!, update_view!, cellplot
# public API with generic names, not exported: call them qualified,
# e.g. `FourierContourCells.live_view(sim)`
public live_view, record_movie

end
