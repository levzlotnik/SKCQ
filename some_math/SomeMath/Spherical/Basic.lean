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
  sorry

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
  sorry

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
  sorry

end Spherical
