module FourierContourCellsEnzymeExt

using Enzyme: Enzyme, Forward, Duplicated, Const, autodiff
using FourierContourCells: EnzymeAD, contour_integrands, energy, bond_energy, as_svector
import FourierContourCells: ad_contour_integrand_gradients, ad_energy_gradient, ad_bond_gradient
using StaticArrays: SVector, hcat

# Forward mode: one pass per input; the inputs are 3 scalars, m integrals or
# the 2 components of a bond vector.

function ad_contour_integrand_gradients(::EnzymeAD, term, R, dR, ddR)
    f(r, r′, r″) = as_svector(contour_integrands(term, r, r′, r″))
    o = one(R)
    gR = only(autodiff(Forward, Const(f), Duplicated(R, o), Const(dR), Const(ddR)))
    gD = only(autodiff(Forward, Const(f), Const(R), Duplicated(dR, o), Const(ddR)))
    gDD = only(autodiff(Forward, Const(f), Const(R), Const(dR), Duplicated(ddR, o)))
    return hcat(gR, gD, gDD)
end

function ad_energy_gradient(::EnzymeAD, term, Q::SVector{m, T}, b) where {m, T}
    F(q) = energy(term, q, b)
    return SVector(ntuple(Val(m)) do j
        e = SVector(ntuple(i -> T(i == j), Val(m)))
        only(autodiff(Forward, Const(F), Duplicated(Q, e)))
    end)
end

function ad_bond_gradient(::EnzymeAD, term, Δ::SVector{2, T}, ti, tj) where {T}
    F(d) = bond_energy(term, d, ti, tj)
    gx = only(autodiff(Forward, Const(F), Duplicated(Δ, SVector(one(T), zero(T)))))
    gy = only(autodiff(Forward, Const(F), Duplicated(Δ, SVector(zero(T), one(T)))))
    return SVector(gx, gy)
end

end
