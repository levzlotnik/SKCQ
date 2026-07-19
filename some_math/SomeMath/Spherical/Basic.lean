import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Analysis.SpecialFunctions.Sqrt

open Real

namespace Spherical

variable {M K D : ℕ}

/-- Squared Euclidean norm. -/
noncomputable def sqNorm (v : Fin D → ℝ) : ℝ := ∑ j : Fin D, v j ^ 2

/-- **Theorem 5 (Spherical k-means objective equivalence).**
On the unit sphere, cosine k-means maximizes Σ_k Σ_{i∈cluster k} ⟨w_i, μ_k⟩
where μ_k is the unit centroid. This is equivalent to minimizing
Σ_k Σ_{i∈k} ‖w_i - μ_k‖² when ‖w_i‖ = ‖μ_k‖ = 1.

Key identity: ‖w - μ‖² = 2 - 2⟨w, μ⟩ on the unit sphere. -/
theorem spherical_kmeans_equivalence
    (W : Fin M → Fin D → ℝ)
    (hW : ∀ i, sqNorm (W i) = 1)
    (μ : Fin K → Fin D → ℝ)
    (hμ : ∀ k, sqNorm (μ k) = 1)
    (cluster : Fin M → Fin K) :
    (∑ i : Fin M, sqNorm (fun j => W i j - μ (cluster i) j)) =
    2 * (M : ℝ) - 2 * (∑ i : Fin M, dotProduct (W i) (μ (cluster i))) := by
  have h_main : ∀ (i : Fin M),
    (∑ j : Fin D, (W i j - μ (cluster i) j) ^ 2) =
    2 - 2 * ∑ j : Fin D, W i j * μ (cluster i) j := by
    intro i
    have h1 : ∑ j : Fin D, W i j ^ 2 = 1 := by
      simpa [sqNorm] using hW i
    have h2 : ∑ j : Fin D, μ (cluster i) j ^ 2 = 1 := by
      simpa [sqNorm] using hμ (cluster i)
    calc
      (∑ j : Fin D, (W i j - μ (cluster i) j) ^ 2)
        = ∑ j : Fin D, (W i j ^ 2 + μ (cluster i) j ^ 2 - 2 * W i j * μ (cluster i) j) := by
          apply Finset.sum_congr rfl
          intro j _
          ring
      _ = ∑ j : Fin D, (W i j ^ 2 + μ (cluster i) j ^ 2) - ∑ j : Fin D, 2 * W i j * μ (cluster i) j := by
        rw [Finset.sum_sub_distrib]
      _ = (∑ j : Fin D, W i j ^ 2 + ∑ j : Fin D, μ (cluster i) j ^ 2) - ∑ j : Fin D, 2 * W i j * μ (cluster i) j := by
        rw [Finset.sum_add_distrib]
      _ = (∑ j : Fin D, W i j ^ 2 + ∑ j : Fin D, μ (cluster i) j ^ 2) - 2 * ∑ j : Fin D, W i j * μ (cluster i) j := by
        simp [Finset.mul_sum, mul_assoc]
      _ = 1 + 1 - 2 * ∑ j : Fin D, W i j * μ (cluster i) j := by
        rw [h1, h2]
      _ = 2 - 2 * ∑ j : Fin D, W i j * μ (cluster i) j := by
        ring
  calc
    (∑ i : Fin M, sqNorm (fun j => W i j - μ (cluster i) j))
      = ∑ i : Fin M, ∑ j : Fin D, (W i j - μ (cluster i) j) ^ 2 := by
        simp [sqNorm]
    _ = ∑ i : Fin M, (2 - 2 * ∑ j : Fin D, W i j * μ (cluster i) j) := by
      apply Finset.sum_congr rfl
      intro i _
      exact h_main i
    _ = ∑ i : Fin M, (2 : ℝ) - ∑ i : Fin M, 2 * ∑ j : Fin D, W i j * μ (cluster i) j := by
      rw [Finset.sum_sub_distrib]
    _ = ∑ i : Fin M, (2 : ℝ) - 2 * ∑ i : Fin M, ∑ j : Fin D, W i j * μ (cluster i) j := by
      rw [Finset.mul_sum]
    _ = 2 * (M : ℝ) - 2 * ∑ i : Fin M, ∑ j : Fin D, W i j * μ (cluster i) j := by
      simp
      ring
    _ = 2 * (M : ℝ) - 2 * (∑ i : Fin M, dotProduct (W i) (μ (cluster i))) := by
      simp [dotProduct]

/-- **Theorem 6 (Scale-refit optimality).**
Given a fixed direction d̂ (from the codebook cascade), the optimal scalar
s* = ⟨w, d̂⟩ / ‖d̂‖² minimizes ‖w - s·d̂‖² over all s ∈ ℝ.
This is the orthogonal projection of w onto the 1D subspace spanned by d̂. -/
theorem scale_refit_optimality
    (w d : Fin D → ℝ)
    (hd : sqNorm d > 0) :
    let s_star := dotProduct w d / sqNorm d
    ∀ s : ℝ,
    sqNorm (fun j => w j - s_star * d j) ≤
    sqNorm (fun j => w j - s * d j) := by
  intro s
  have h₁ : 0 < ∑ j : Fin D, d j ^ 2 := by simpa [sqNorm] using hd
  have h₂ : ∑ j : Fin D, (w j - s * d j) ^ 2 - ∑ j : Fin D, (w j - (dotProduct w d / ∑ j : Fin D, d j ^ 2) * d j) ^ 2 ≥ 0 := by
    have h₃ : ∑ j : Fin D, (w j - s * d j) ^ 2 - ∑ j : Fin D, (w j - (dotProduct w d / ∑ j : Fin D, d j ^ 2) * d j) ^ 2 =
        (s - dotProduct w d / ∑ j : Fin D, d j ^ 2) ^ 2 * ∑ j : Fin D, d j ^ 2 := by
      calc
        _ = ∑ j : Fin D, ((w j - s * d j) ^ 2 - (w j - (dotProduct w d / ∑ j : Fin D, d j ^ 2) * d j) ^ 2) := by
          rw [Finset.sum_sub_distrib]
        _ = ∑ j : Fin D, (2 * (dotProduct w d / ∑ j : Fin D, d j ^ 2 - s) * w j * d j +
              (s ^ 2 - (dotProduct w d / ∑ j : Fin D, d j ^ 2) ^ 2) * d j ^ 2) := by
          apply Finset.sum_congr rfl
          intro j _
          ring
        _ = 2 * (dotProduct w d / ∑ j : Fin D, d j ^ 2 - s) * ∑ j : Fin D, w j * d j +
              (s ^ 2 - (dotProduct w d / ∑ j : Fin D, d j ^ 2) ^ 2) * ∑ j : Fin D, d j ^ 2 := by
          rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.sum_mul]
        _ = 2 * (dotProduct w d / ∑ j : Fin D, d j ^ 2 - s) * dotProduct w d +
              (s ^ 2 - (dotProduct w d / ∑ j : Fin D, d j ^ 2) ^ 2) * ∑ j : Fin D, d j ^ 2 := by
          simp [dotProduct]
        _ = (s - dotProduct w d / ∑ j : Fin D, d j ^ 2) ^ 2 * ∑ j : Fin D, d j ^ 2 := by
          field_simp [h₁.ne']
          ring
    rw [h₃]
    apply mul_nonneg
    · exact sq_nonneg _
    · exact Finset.sum_nonneg fun j _ => sq_nonneg _
  linarith

/-- **Theorem 7 (Unit-sphere residual orthogonality).**
After normalizing w to ŵ = w/‖w‖ and subtracting the projection onto the assigned
centroid μ (where ‖μ‖ = 1), the residual r = ŵ - ⟨ŵ, μ⟩·μ is orthogonal to μ. -/
theorem unit_sphere_residual_orthogonality
    (w μ : Fin D → ℝ)
    (hw : sqNorm w > 0)
    (hμ : sqNorm μ = 1) :
    let ŵ : Fin D → ℝ := fun j => w j / Real.sqrt (sqNorm w)
    let r : Fin D → ℝ := fun j => ŵ j - dotProduct ŵ μ * μ j
    dotProduct r μ = 0 := by
  dsimp only
  have h₁ : ∑ j : Fin D, μ j * μ j = 1 := by
    simpa [sqNorm, pow_two] using hμ
  have h₂ : ∑ j : Fin D, (w j / Real.sqrt (sqNorm w)) * μ j =
      (1 / Real.sqrt (sqNorm w)) * ∑ j : Fin D, w j * μ j := by
    have : ∀ j, (w j / Real.sqrt (sqNorm w)) * μ j = (1 / Real.sqrt (sqNorm w)) * (w j * μ j) := by
      intro j; ring
    simp only [this, Finset.sum_mul]
  calc
    dotProduct (fun j => w j / Real.sqrt (sqNorm w) - (∑ j : Fin D, w j / Real.sqrt (sqNorm w) * μ j) * μ j) μ
      = ∑ j : Fin D, (w j / Real.sqrt (sqNorm w) - (∑ j : Fin D, w j / Real.sqrt (sqNorm w) * μ j) * μ j) * μ j := by
        simp [dotProduct]
      _ = ∑ j : Fin D, ((w j / Real.sqrt (sqNorm w)) * μ j - (∑ j : Fin D, w j / Real.sqrt (sqNorm w) * μ j) * μ j * μ j) := by
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = ∑ j : Fin D, (w j / Real.sqrt (sqNorm w)) * μ j - ∑ j : Fin D, (∑ j : Fin D, w j / Real.sqrt (sqNorm w) * μ j) * (μ j * μ j) := by
        rw [Finset.sum_sub_distrib]
      _ = ∑ j : Fin D, (w j / Real.sqrt (sqNorm w)) * μ j - (∑ j : Fin D, w j / Real.sqrt (sqNorm w) * μ j) * ∑ j : Fin D, μ j * μ j := by
        rw [Finset.sum_mul]
      _ = ∑ j : Fin D, (w j / Real.sqrt (sqNorm w)) * μ j - (∑ j : Fin D, w j / Real.sqrt (sqNorm w) * μ j) * 1 := by
        rw [h₁]
      _ = 0 := by
        ring

end Spherical
