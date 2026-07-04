import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Analysis.SpecialFunctions.Sqrt

namespace ErrorProp

variable {M D E : ℕ}

/-- Squared Euclidean norm. -/
noncomputable def sqNorm (v : Fin D → ℝ) : ℝ := ∑ j : Fin D, v j ^ 2

/-- Euclidean norm. -/
noncomputable def myNorm (v : Fin D → ℝ) : ℝ := Real.sqrt (sqNorm v)

/-- **Theorem 8 (Per-token output bound).**
Expert layer computes y = x·W^T. With quantized Ŵ, ŷ = x·Ŵ^T.
Then ‖y - ŷ‖ ≤ ‖x‖ · ‖W - Ŵ‖_F (Cauchy-Schwarz on the matvec).

Here we express this for a single expert: if y_i = Σ_j x_j · W_{ij} and
ŷ_i = Σ_j x_j · Ŵ_{ij}, then the output error is bounded by ‖x‖ · ‖W - Ŵ‖_F. -/
theorem per_token_output_bound
    (x : Fin D → ℝ)
    (W Ŵ : Fin M → Fin D → ℝ) :
    (∑ i : Fin M, (∑ j : Fin D, x j * (W i j - Ŵ i j)) ^ 2) ≤
    sqNorm x * (∑ i : Fin M, ∑ j : Fin D, (W i j - Ŵ i j) ^ 2) := by
  sorry

/-- **Theorem 9 (PQ per-token bound).**
‖y - ŷ‖ ≤ ‖x‖ · √(Σ_b ‖W_b - Ŵ_b‖_F²) — directly from the PQ Frobenius decomposition.

Expressed in block form: the output error is bounded by the input norm times the
square root of the sum of per-block squared Frobenius errors. -/
theorem pq_per_token_bound
    (B : ℕ) (hB : B > 0)
    (x : Fin D → ℝ)
    (W Ŵ : Fin M → Fin B → Fin (D / B) → ℝ)
    (hblock : D % B = 0) :
    (∑ i : Fin M, (∑ b : Fin B, ∑ j : Fin (D / B), x ⟨b.val * (D / B) + j.val, by
      have hdiv : D / B * B = D := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hblock)
      have hb : b.val < B := Fin.prop b
      have hj : j.val < D / B := Fin.prop j
      have : b.val * (D / B) + j.val < B * (D / B) := by
        nlinarith
      linarith
    ⟩ * (W i b j - Ŵ i b j)) ^ 2) ≤
    sqNorm x * (∑ i : Fin M, ∑ b : Fin B, ∑ j : Fin (D / B), (W i b j - Ŵ i b j) ^ 2) := by
  sorry

/-- **Theorem 10 (Routing sensitivity).**
If the router selects top-k experts with weights g_1, ..., g_k, the total output error is
‖ŷ - y‖ ≤ Σ_{e∈top-k} |g_e| · ‖x‖ · ‖W_e - Ŵ_e‖_F.

Expressed here for a mixture-of-experts layer with routing weights. -/
theorem routing_sensitivity
    (x : Fin D → ℝ)
    (g : Fin E → ℝ)
    (W Ŵ : Fin E → Fin M → Fin D → ℝ) :
    (∑ i : Fin M, (∑ e : Fin E, g e * ∑ j : Fin D, x j * (W e i j - Ŵ e i j)) ^ 2) ≤
     (∑ e : Fin E, |g e| * Real.sqrt (sqNorm x * (∑ i : Fin M, ∑ j : Fin D,
      (W e i j - Ŵ e i j) ^ 2))) ^ 2 := by
  sorry

end ErrorProp
