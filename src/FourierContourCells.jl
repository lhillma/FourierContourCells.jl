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

# geometry and state
export FourierBasis, PeriodicBox, Bodies, BodyType, nbodies, ncoef, nmodes

end
