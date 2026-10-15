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

# geometry and state
export FourierBasis, PeriodicBox, Bodies, BodyType, nbodies, ncoef, nmodes
# energy interface
export EnergyTerm, BodyTerm, ContactTerm, BodyInfo, OnTypes
export densities, combine, density_gradients, combine_gradient, pressure, interaction_range,
    applies_to, ad_backend, ForwardDiffAD, EnzymeAD
# built-in energy terms
export Tension, PerimeterElasticity, Overlap, LinearRepulsion
# simulation
export Simulation, step!, run!, compute_gradients!, hexagonal_monolayer, CPU
# observables
export perimeters, areas, shape_index, positions, unwrapped_positions, contours, body_energy

end
