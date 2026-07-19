import Mathlib.Analysis.Matrix.Normed
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Real.Basic

open scoped Matrix.Norms.Frobenius

variable {M B N K : ℕ}

/-- Product quantization reconstruction: given codebooks and assignments,
reconstruct the quantized weight tensor. -/
def pqReconstruct (C : Fin B → Fin K → Fin N → ℝ)
    (π : Fin M → Fin B → Fin K) (i : Fin M) (b : Fin B) (j : Fin N) : ℝ :=
  C b (π i b) j

/-- The squared Frobenius norm of the PQ error, expressed as a triple sum. -/
def pqErrorSquared (W : Fin M → Fin B → Fin N → ℝ)
    (C : Fin B → Fin K → Fin N → ℝ) (π : Fin M → Fin B → Fin K) : ℝ :=
  ∑ i : Fin M, ∑ b : Fin B, ∑ j : Fin N, (W i b j - pqReconstruct C π i b j) ^ 2

/-- **Theorem 1 (PQ Frobenius decomposition).**
Given a weight tensor W split into B blocks of size N, the squared Frobenius norm
of the quantization error equals the sum of per-block squared errors. -/
theorem pq_frobenius_decomposition
    (W : Fin M → Fin B → Fin N → ℝ)
    (C : Fin B → Fin K → Fin N → ℝ)
    (π : Fin M → Fin B → Fin K) :
    pqErrorSquared W C π =
    ∑ b : Fin B, ∑ i : Fin M, ∑ j : Fin N, (W i b j - C b (π i b) j) ^ 2 := by
  simp [pqReconstruct, pqErrorSquared]
  rw [Finset.sum_comm]
  rfl

/-- **Theorem 2 (Optimal assignment bound).**
For fixed codebooks, the optimal per-row assignment minimizes each block term independently.
The total squared error under any assignment π is at least the sum over rows and blocks of the
minimum squared distance to any codebook entry. -/
theorem pq_optimal_assignment_bound
    (W : Fin M → Fin B → Fin N → ℝ)
    (C : Fin B → Fin K → Fin N → ℝ)
    (π : Fin M → Fin B → Fin K) :
    pqErrorSquared W C π ≥
    ∑ i : Fin M, ∑ b : Fin B, ⨅ k : Fin K, ∑ j : Fin N, (W i b j - C b k j) ^ 2 := by
  simp [pqReconstruct, pqErrorSquared]
  refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun b _ => ?_
  refine le_ciInf ?_ (Finset.mem_univ _)
  intro k _
  exact Finset.single_le_sum (fun j _ => sq_nonneg _) (Finset.mem_univ k)
