# Cells with a nucleus: modelling notes

This is an extension of the Fourier contour cell model of N. Saito and S. Ishihara,
Sci. Adv. 10, eadi8433 (2024), doi:10.1126/sciadv.adi8433. It covers the choices
behind `examples/nucleus.jl` and the measurements that support them. Run the
checks with

```sh
julia -t auto --project=examples examples/nucleus.jl test    # pull-in check (CPU, Float64)
julia -t auto --project=examples examples/nucleus.jl check Jw=2 k=0.2 drag=parent ηn=0.05
```

## Model

Every cell (type 1) contains one nucleus (type 2). The nucleus is a Fourier
contour body of its own whose `parent` is the cell (`with_nuclei(cells; ratio
= 0.5)`). Both have hard area constraints, so the nucleus area is π(0.5 R₀)².

| ingredient | choice | library feature |
|:--|:--|:--|
| excluded volume | overlap with any other cell or nucleus costs J = 1 per unit area | `Overlap(J)` |
| confinement | a cell and its own nucleus *gain* Jw = 2 per unit overlap area | negative `related` coupling of `Overlap` |
| activity | the nucleus has v0 = 0; its θ diffuses (Dr = 0.01) but only sets the frame of its shape modes | `BodyType(v0 = 0)` |
| anchoring | harmonic bond (k/2) \|r_n − r_c\|² between the centres, k = 0.2 | `HarmonicBond`, `parent_bonds` |
| drag | relative to the cell (cytoplasm), not the substrate | `BodyType(drag = :parent)` |
| stiffness | line tension 0.05 on nuclei vs 0.01 on cells | `OnTypes(Tension(η), type)` |

### Confinement by the wrap coupling

With `related = 0` the membrane does not feel its own nucleus. Elongating cells
then narrow past it, and about ten nuclei per 1024 cells poke out by up to
0.04 R₀ at t = 100. A negative `related` overlap coupling −Jw rewards overlap
of a cell with its own nucleus. The membrane then wraps the nucleus, and
parts that poke out are pulled back.

This does not cause wetting. The overlap is at most the fixed nucleus area,
and a nucleus fully inside its cell feels no force from it. The `test` mode
starts a soft nucleus (ηn = 0.005, no bond) 0.2 R₀ outside its cell. The
nucleus is pulled in by t ≈ 2, and its modes stay ≤ 0.005 and fall off with
n. The cell bulges briefly (|c₂| ≈ 0.014) to wrap the nucleus, and its line
tension relaxes the bulge again. Once inside, the nucleus does not stick to
the membrane.

Jw = 2 > J makes the wrap win against neighbours pressing on the cell. The
alternative, an energy for the nucleus area outside its cell, would need an
interaction range of about a nucleus diameter for all pairs and thus much
longer neighbour lists.

### Drag relative to the cell

The nucleus feels drag against the cytoplasm, i.e. relative to its cell. The
cell bears the nucleus's forces and carries it along:

  ṙ_c = v0 e(θ_c) + μr_c (F_c + F_n),  ṙ_n = ṙ_c + μr_n F_n.

The bond is then an internal force. It only re-centres the nucleus at rate
μr_n k and does not slow the cell down; an isolated cell moves at v0 without
lag.

Substrate drag on the nucleus would double-count the friction of the cell's
footprint. An isolated pair with μr = 1 for both bodies would move at v0/2,
independent of k, with a lag v0/(2 μr k) (0.15 R₀ at k = 0.2).

### Anchoring

The spring acts on the polar origins r of the contours. These differ from
the area centroids once modes n ≥ 2 are excited, because products of modes
create n = 1 harmonics in ∮ R³ e_φ dφ. This is accepted for now, since the
nuclei stay nearly circular.

## Measurements

Monolayer of 1024 cells with nuclei, packing 0.95, soft fluid cell parameters
(v0 = 0.06, Dr = 0.01, η = 0.01, μab = 0.1), dt = 0.005, t = 100 (20 000
steps). Each row is one run with the same seed, on an RTX A2000 GPU in
Float32 (`examples/nucleus.jl check`). Columns:

- protruding: number of nuclei poking out of their cell;
- max protrusion: largest protrusion in R₀ (negative: all inside, with margin);
- ⟨|c₂|⟩, ⟨|c₆|⟩: mean nucleus mode amplitudes;
- p_n max: largest nucleus shape index (circle 2√π ≈ 3.545);
- Δr: mean cell displacement;
- ⟨p⟩: mean cell shape index;
- offset: mean nucleus–cell centre distance.

Wrap coupling and bond (relative drag):

| Jw | k | ηn | protruding | max protrusion | ⟨\|c₂\|⟩ | ⟨\|c₆\|⟩ | p_n max | Δr | ⟨p⟩ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.05 | 10 | 0.041 | 6.4e-5 | 9.9e-6 | 3.55 | 0.933 | 3.82 |
| 2 | 1 | 0.05 | 0 | −0.0007 | 5.8e-5 | 6.7e-6 | 3.55 | 0.932 | 3.82 |
| 0 | 0.2 | 0.05 | 9 | 0.026 | 2.8e-5 | 3.9e-6 | 3.55 | 0.932 | 3.82 |
| **2** | **0.2** | **0.05** | 0 | −0.0017 | 2.6e-5 | 5.3e-6 | 3.55 | 0.924 | 3.82 |
| 2 | 0.2 | 0.005 | 0 | −0.0002 | 3.7e-5 | 1.6e-5 | 3.55 | 0.936 | 3.82 |
| 5 | 0.2 | 0.005 | 0 | −0.0029 | 3.9e-5 | 8.6e-6 | 3.55 | 0.909 | 3.82 |

The wrap coupling removes all protrusions. Cell shapes and displacements do
not change within noise, and the nuclei stay circular even with a ten times
softer line tension: other bodies rarely touch them.

Drag (Jw = 2, k = 0.2):

| | Δr | ⟨p⟩ | offset |
|:--|---:|---:|---:|
| cells only, no nuclei | 0.923 | 3.82 | |
| nuclei, relative drag | 0.924 | 3.82 | 0.0002 |
| nuclei, substrate drag | 0.594 | 3.76 | 0.040 |

With relative drag the monolayer matches the cells-only model. Substrate drag
slows the cells down by a third.

Cost: the nuclei double the number of bodies and enter the neighbour lists of
nearby cells. The contact kernel also evaluates every cell–own-nucleus pair,
which the wrap coupling needs. A step takes 0.16 ms on the GPU (Float32, CUDA
graphs) vs 0.08 ms for the cells alone, and 2.3 ms on 16 CPU threads
(Float64). GPU and CPU statistics agree (Δr 0.93 vs 0.94, ⟨p⟩ 3.819 vs 3.823,
no protruding nuclei).

## Open points

- The bond acts on the centres, not the centroids (see Anchoring).
- All body types share the number of shape modes and contour nodes. Nuclei
  could use fewer of both.
- Energy diagnostics: `body_energy` sums the single-body terms only. Bond
  and contact energies are never evaluated, since the dynamics needs only
  their gradients. A bond energy observable would be cheap; a contact energy
  needs an area integral (radial quadrature per node). Both belong with the
  analysis tools, not this model.
- The cost of the nucleus pairs in the contact kernel (see Cost).
