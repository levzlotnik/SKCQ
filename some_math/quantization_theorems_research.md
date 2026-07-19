# Mathematical Theorems Behind Quantization: Exact Statements from the Literature

## Research compiled: July 2026

---

## 1. ZADOR'S THEOREM (Classical K-Means Rate-Distortion)

### 1.1 The Exact Theorem Statement

**Reference:** Graf & Luschgy, "Foundations of Quantization for Probability Distributions", Lecture Notes in Mathematics 1730, Springer, 2000, Theorem 6.2.

Also see: Luschgy & Pages (2023), "Greedy vector quantization", J. Approx. Theory 198.

**Theorem (Zador, Graf & Luschgy 2000):**

Let r > 0 and let ||.|| denote any norm on R^d. Let P be a probability measure on (R^d, Borel(R^d)) with density h = dP^a/d(lambda_d) of the absolutely continuous part P^a with respect to Lebesgue measure lambda_d.

Assume the moment condition:
  integral_{R^d} ||xi||^{r+delta} P(d xi) < +infinity  for some delta > 0.

Then:

  lim_{n -> infinity} n^{1/d} e_{n,r}(P, ||.||) = Q_r([0,1]^d, ||.||)^{1/r} * ||h||_{L^{d/(d+r)}(lambda_d)}^{1/r}

where:
- e_{n,r}(P, ||.||) = inf_{|Gamma| <= n} [ integral min_{a in Gamma} ||xi - a||^r P(d xi) ]^{1/r}
  is the L^r-optimal mean quantization error at level n.
- Q_r([0,1]^d, ||.||) := inf_{n >= 1} n^{r/d} e_{r,n}([0,1]^d, ||.||)^r
  = lim_{n >= 1} n^{r/d} e_{r,n}([0,1]^d, ||.||)^r in (0, +infinity)
  is the quantization coefficient for the uniform distribution on the unit cube.

**Lower bound (always true, no moment condition needed):**
  liminf_{n -> infinity} n^{1/d} e_{n,r}(P, ||.||) >= Q_r([0,1]^d, ||.||)^{1/r} * ||h||_{L^{d/(d+r)}(lambda_d)}^{1/r}

### 1.2 Precise Hypotheses

1. P must be a probability measure on R^d (Borel sigma-algebra)
2. The (r+delta)-moment must be finite for some delta > 0
3. h is the density of the absolutely continuous part of P w.r.t. Lebesgue measure
4. The norm ||.|| can be ANY norm on R^d (not just Euclidean)
5. r > 0 (the result holds for all positive r, not just r >= 1)

### 1.3 The Rate

The rate is Theta(n^{-1/d}), i.e., the quantization error decays as n^{-1/d}.
Equivalently, the distortion (r-th power of error) decays as n^{-r/d}.

### 1.4 History

- **Zador (1966)**: PhD thesis, Bell Labs Tech Memo. Original but incomplete proof.
- **Bucklew & Wise (1982)**: Extended to distributions with enough finite moments, but had a gap.
- **Graf & Luschgy (2000)**: First fully rigorous proof, monograph [GL00, Thm. 6.2].
- **Luschgy & Pages (2023)**: Showed the result holds for r in (0, infinity), not just r >= 1.
- **Aydin (2025)**: Extended to rectifiable measures on R^d, answered Graf-Luschgy conjecture.

### 1.5 Known Values of Q_r

- d=1: Q_r([0,1], |.|) = 1 / (2^r * (r+1)^{1/r})
- d=2, r=2: Q_2([0,1]^2) = 5/(18*sqrt(3)) (hexagonal lattice)
- d=3, r=2: Q_2([0,1]^3) = 19/(64 * 2^{1/3}) (conjectured, truncated octahedron)
- Gersho's conjecture (1979): For each d, there exists a lattice achieving Q_{2,d}. OPEN for d >= 3.

### 1.6 Common Misconceptions

**WRONG:** "Zador's theorem says the k-means distortion is Theta(K^{-2/d})"
**CORRECT:** Zador's theorem gives the EXACT asymptotic constant, not just Theta. The distortion V_n(P) = e_{n,2}(P)^2 satisfies:
  lim_{n->inf} n^{2/d} V_n(P) = Q_2([0,1]^d)^{2/2} * ||h||_{L^{d/(d+2)}}^{2/2}
  = Q_2([0,1]^d) * ||h||_{L^{d/(d+2)}}

**WRONG:** "Zador proved this theorem completely"
**CORRECT:** Zador's original proof was incomplete. The first complete proof is due to Graf & Luschgy (2000).

**WRONG:** "The result requires r >= 1"
**CORRECT:** A careful reading of Graf-Luschgy's proof shows it holds for all r > 0.

### 1.7 Lean 4 Formalization Suggestion

```lean4
/-- The L^r optimal quantization error at level n for distribution P -/
def quantizationError (r : ℝ) (P : Measure ℝ^d) (n : ℕ) : ℝ :=
  ⨅ (Γ : Finset (ℝ^d)) (h : Γ.card ≤ n),
    (∫ x, (⨅ a ∈ Γ, ‖x - a‖ ^ r) ∂P) ^ (1/r)

/-- Zador's constant for the unit cube -/
noncomputable def zadorConstant (r : ℝ) (d : ℕ) : ℝ :=
  ⨅ n : ℕ, (n : ℝ) ^ (r/d) * (quantizationError r (uniformOnUnitCube d) n) ^ r

/-- Zador's Theorem: asymptotic rate of optimal quantization -/
theorem zador_theorem
  (r : ℝ) (hr : r > 0)
  (P : Measure ℝ^d)
  (h_moment : ∃ δ > 0, ∫ x, ‖x‖ ^ (r + δ) ∂P < ∞)
  (h : ℝ^d → ℝ) (h_density : h = RadonNikodym.deriv P.absolutelyContinuousPart volume) :
  Filter.Tendsto (fun n : ℕ ↦ (n : ℝ) ^ (1/d : ℝ) * quantizationError r P n)
    Filter.atTop (𝓝 (zadorConstant r d ^ (1/r) * ‖h‖_(d/(d+r)) ^ (1/r))) :=
```

---

## 2. SPHERICAL K-MEANS RATE-DISTORTION

### 2.1 The Exact Rate

For a probability measure P on the unit sphere S^{d-1} subset R^d, the quantization error with respect to geodesic distance (or equivalently, chord distance) decays at rate:

  V_n(P) = Theta(n^{-2/(d-1)})

**Why d-1?** Because S^{d-1} is a (d-1)-dimensional Riemannian manifold. Zador's theorem on a d-dimensional Riemannian manifold gives rate n^{-1/d}; on S^{d-1} (dimension d-1), this becomes n^{-1/(d-1)} for the error, or n^{-2/(d-1)} for the squared distortion.

### 2.2 Precise Statement for S^{d-1}

**Theorem (Graf & Luschgy 2000, Thm. 6.2 applied to manifolds):**

Let P be a probability measure on S^{d-1} with density h w.r.t. the (d-1)-dimensional Hausdorff measure H^{d-1} on S^{d-1}. If P has appropriate integrability (finite (r+delta)-moment for some delta > 0, which is automatic on the compact sphere), then:

  lim_{n -> infinity} n^{1/(d-1)} e_{n,r}(P, d_G) = Q_r^{S^{d-1}} * ||h||_{L^{(d-1)/((d-1)+r)}(H^{d-1})}^{1/r}

where d_G(x,y) = arccos(<x,y>) is the geodesic distance on S^{d-1}.

### 2.3 For the 1-Dimensional Case (Great Circles)

**Theorem (Roychowdhury 2025, arXiv:2511.05099, Theorem 4.5):**

Let P be the uniform probability distribution on a great circle (equator) Gamma of the sphere S^2 with total geodesic length L = 2*pi. For squared geodesic distortion:

  V_n(P) = L^2 / (12 * n^2) = pi^2 / (3 * n^2)

More generally, for any uniform distribution on a 1-dimensional geodesic subset of total length L:

  V_n = L^2 / (12 * n^2)

### 2.4 References

- Graf & Luschgy (2000), "Foundations of Quantization...", Springer LNM 1730
  - Thm. 6.2 for Euclidean; reduction to manifolds via charts (Gru04, Klo12, Iac16)
- Roychowdhury (2025), "Optimal Quantization on Spherical Surfaces", arXiv:2511.05099
  - Explicit formulas for 1-D spherical subsets
- Iacobelli (2016), "Quantization for probability measures on Riemannian manifolds"
  - Zador's theorem on compact Riemannian manifolds via chart reduction

### 2.5 Common Misconceptions

**WRONG:** "Spherical k-means has rate Theta(K^{-2/d})"
**CORRECT:** The rate is Theta(K^{-2/(d-1)}) because the sphere S^{d-1} is (d-1)-dimensional.

**WRONG:** "The spherical case is fundamentally different from Euclidean"
**CORRECT:** For compact manifolds, Zador's theorem reduces to the Euclidean case via local charts. The constant changes (depends on the manifold geometry) but the rate n^{-1/(dim)} is the same.

### 2.6 Lean 4 Formalization Suggestion

```lean4
/-- Geodesic distance on the unit sphere -/
def geodesicDist (x y : Sphere (d-1)) : ℝ := Real.arccos (inner x y)

/-- Spherical quantization error -/
def sphericalQuantError (r : ℝ) (P : Measure (Sphere (d-1))) (n : ℕ) : ℝ :=
  ⨅ (Γ : Finset (Sphere (d-1))) (h : Γ.card ≤ n),
    (∫ x, (⨅ a ∈ Γ, geodesicDist x a ^ r) ∂P) ^ (1/r)

/-- Spherical Zador theorem -/
theorem spherical_zador_theorem
  (r : ℝ) (hr : r > 0)
  (P : Measure (Sphere (d-1)))
  (h : Sphere (d-1) → ℝ) (h_density : h = RadonNikodym.deriv P (hausdorffMeasure (d-1))) :
  Filter.Tendsto (fun n : ℕ ↦ (n : ℝ) ^ (1/(d-1 : ℝ)) * sphericalQuantError r P n)
    Filter.atTop (𝓝 (sphericalZadorConst r d * ‖h‖_((d-1)/((d-1)+r)) ^ (1/r))) :=
```

---

## 3. UNIT SPHERE COVERING / QUANTIZATION

### 3.1 Covering Numbers of the Sphere

**Definition:** The covering number N(S^{d-1}, epsilon) is the minimum number of spherical caps of angular radius epsilon needed to cover S^{d-1}.

**Volume argument (standard):**
  N(S^{d-1}, epsilon) <= C_d * (1/epsilon)^{d-1}

where C_d depends on dimension. More precisely, for small epsilon:

  N(S^{d-1}, epsilon) ~ (1/epsilon)^{d-1} * (surface area ratio)

### 3.2 Rogers Bound for Covering Density

**Theorem (Rogers 1963):**

For covering R^d (or S^d) with balls of radius r, the covering density theta satisfies:
  theta <= (1 + o(1)) * d * ln(d)  as d -> infinity

This was recently improved (Dumer 2006, math/0606002):
  theta <= (1 + o(1)) * (d * ln(d)) / 2  as d -> infinity

### 3.3 Relation Between Covering and K-Means Distortion

The covering radius of a set S of n points on S^{d-1} is:
  rho(S) = max_{x in S^{d-1}} min_{s in S} d_G(x, s)

This is the L^infinity quantization error. The L^2 quantization error (k-means distortion) satisfies:
  V_n(P) <= rho(S)^2

For the uniform distribution on S^{d-1}:
  V_n ~ C * n^{-2/(d-1)}

### 3.4 Key References

- Rogers (1963): "Covering a sphere with spheres", Mathematika
- Dumer (2006): "Covering spheres with spheres", arXiv:math/0606002
  - Improved Rogers bound by factor of 2: theta ~ (d ln d)/2
- Gao (2026): "New upper bound for lattice covering by spheres", Mathematika
  - Further improvement for lattice coverings
- Campos, Jenssen, Michelen, Sahasrabudhe (2024): "A new lower bound for sphere packing"
  - arXiv:2312.10026, first asymptotic improvement on Rogers' packing bound

### 3.5 Lean 4 Formalization Suggestion

```lean4
/-- Covering number of the sphere -/
def sphereCoveringNumber (d : ℕ) (epsilon : ℝ) : ℕ :=
  sInf {n : ℕ | ∃ (S : Finset (Sphere (d-1))), S.card ≤ n ∧
    ∀ x : Sphere (d-1), ∃ s ∈ S, geodesicDist x s ≤ epsilon}

/-- Rogers bound on covering density -/
theorem rogers_covering_bound (d : ℕ) (hd : d ≥ 3) :
  ∃ C : ℝ, ∀ epsilon > 0,
    sphereCoveringNumber d epsilon ≤ C * (1/epsilon)^(d-1) * d * Real.log d :=
```

---

## 4. GREEDY RESIDUAL QUANTIZATION / ADDITIVE QUANTIZATION

### 4.1 Greedy Vector Quantization

**Theorem (Luschgy & Pages 2015, J. Approx. Theory 198):**

Let X be an R^d-valued random vector with X in L^p (i.e., E||X||^p < infinity).

Define the greedy sequence (a_N)_{N>=1} recursively:
  a_1 = argmin_a E||X - a||^p  (L^p-median)
  a_N = argmin_a E[min(||X - a_1||, ..., ||X - a_{N-1}||, ||X - a||)^p]

Then the greedy sequence is L^p-rate optimal:
  e_p(a^{(N)}, X) = O(N^{-1/d})  as N -> infinity

where a^{(N)} = (a_1, ..., a_N).

**Distortion mismatch property:** Under additional assumptions, the N-tuples a^{(N)} remain rate optimal w.r.t. L^q norms for p <= q < p + d.

### 4.2 Residual Vector Quantization (RVQ)

RVQ represents x approximately as a sum of codewords:
  x ≈ c_1(i_1) + c_2(i_2) + ... + c_M(i_M)

where c_m(i_m) is selected from codebook C_m by quantizing the residual r_{m-1} = x - sum_{j<m} c_j(i_j).

**Key fact:** The encoding problem is NP-hard in general (Liu et al. 2015, 2016). The standard greedy approach selects at each stage:
  c_m = argmin_{c in C_m} ||r_{m-1} - c||

This greedy encoding is suboptimal but runs in O(M * K) time (M stages, K codewords each).

### 4.3 Additive Quantization (AQ)

**Reference:** Babenko & Lempitsky (2014), "Additive Quantization for Extreme Vector Compression", CVPR 2014.

AQ represents x as sum of M codewords from M different codebooks:
  x ≈ sum_{m=1}^{M} c_m(i_m),  c_m(i_m) ∈ C_m

**No theoretical approximation guarantee exists** for AQ or PQ in terms of a multiplicative constant relative to the optimal quantizer. This is explicitly noted in:
- RaBitQ paper (Gao & Long, 2024): "none of PQ and its variants provide a theoretical error bound"
- The encoding problem is NP-hard, so no polynomial-time algorithm can achieve a constant-factor approximation unless P=NP.

### 4.4 Connection to Submodular Optimization

The set-function f(S) = -E[min_{a in S} ||X - a||^2] is NOT submodular in general. However, for certain formulations:

- The coverage function f(S) = E[min_{a in S} ||X - a||^2] is supermodular (the negative is submodular)
- Greedy maximization of monotone submodular functions gives (1 - 1/e)-approximation (Nemhauser et al. 1978)
- For non-submodular functions with submodularity ratio gamma: greedy gives (1 - e^{-gamma})-approximation (Das & Kempe 2011)

**However**, the k-means objective is NOT submodular, so these guarantees do NOT directly apply to k-means or quantization.

### 4.5 What IS Known About Greedy Quantization Error

**Theorem (Luschgy & Pages, extended by El Nmeir, Luschgy, Pages 2020):**

For an L^r-optimal greedy sequence (a_n) for a distribution P on R^d satisfying appropriate conditions:
  e_{r}(a^{(n)}, P) = O(n^{-1/d})

This is the SAME rate as optimal quantizers (Zador's theorem), but with a potentially worse constant. The greedy sequence is "rate optimal" but not necessarily asymptotically optimal.

### 4.6 Common Misconceptions

**WRONG:** "Greedy additive quantization has O(log K) approximation guarantee"
**CORRECT:** There is NO constant-factor approximation guarantee for additive quantization. The encoding problem is NP-hard. Greedy gives the correct asymptotic RATE O(n^{-1/d}) but with an unknown/worse constant.

**WRONG:** "Submodular optimization applies to k-means"
**CORRECT:** The k-means objective is NOT submodular. Submodular guarantees (1-1/e etc.) do not apply.

### 4.7 Lean 4 Formalization Suggestion

```lean4
/-- Greedy quantization sequence -/
def greedyQuantizer (P : Measure ℝ^d) : ℕ → ℝ^d
  | 0 => Classical.choose (exists_min_Lp_median P)
  | n+1 => Classical.choose (exists_min_step P (greedyQuantizer P n))

/-- Greedy quantization is rate-optimal -/
theorem greedy_quantization_rate_optimal
  (P : Measure ℝ^d) (r : ℝ) (hr : r > 0)
  (h_moment : ∫ x, ‖x‖^r ∂P < ∞)
  (h_regularity : P.satisfiesControlCondition) :
  ∃ C : ℝ, ∀ n : ℕ, n ≥ 1 →
    (∫ x, (⨅ i < n, ‖x - greedyQuantizer P i‖^r) ∂P)^(1/r) ≤ C * n^(-1/d : ℝ) :=
```

---

## 5. PQ ERROR DECOMPOSITION AND MoE QUANTIZATION ERROR

### 5.1 Product Quantization Error

PQ decomposes R^D into M subspaces of dimension D/M each. The quantization error is:
  E_PQ = sum_{m=1}^{M} E_m

where E_m is the quantization error in subspace m.

**No theoretical error bound exists** for PQ (RaBitQ paper, 2024). PQ uses k-means in each subspace, which is a heuristic with no guaranteed approximation ratio.

### 5.2 MoE Quantization Error Propagation

**Theorem (Dar 2025, arXiv:2510.03151, Theorem 5.1):**

For a MoE model with m experts partitioning [0,1]^d into regions {A_i}, with regression function beta and input density p_x:

The approximation error satisfies:
  E_app <= sigma_epsilon^2 + d * sum_{i=1}^{m} ||nabla beta(x_i)||^2 * p_x(x_i) * M(A_i) * V(A_i)^{1+2/d} + o(V_max^{1+2/d})

where:
- M(A_i) is the normalized second moment of inertia of region A_i
- V(A_i) is the volume of region A_i
- x_i is the center of region A_i

**Corollary (Dar 2025, Corollary 5.3):**

Under appropriate assumptions:
  E_app(H_{m,d}^c) = sigma_epsilon^2 + O(d / m^{2/d})

This shows the approximation error decays as m^{-2/d} with the number of experts m, matching the Zador rate for d-dimensional quantization.

### 5.3 Error Propagation in Quantized MoE Layers

For a quantized MoE layer with up-projection W_u and down-projection W_d:

**Error bound (from ACL 2025 findings paper):**

  ||Delta y||^2 <= kappa(W_d) * epsilon_b(W_d)^2 * ||x||^2 + ||W_d||^2 * epsilon_b(W_u)^2 * ||x||^2

where:
- kappa(W_d) is the condition number of W_d
- epsilon_b(W) is the quantization error of weight matrix W at bit-width b

### 5.4 Routing Error Under Quantization

Quantization perturbations in MoE cause TWO types of errors:
1. **Numerical error**: Standard weight quantization error in expert parameters
2. **Expert-shift (misrouting)**: Quantization noise in router logits changes top-k expert selection

The expert-shift error is fundamentally different from numerical error because it changes the computation path. This is studied in:
- MoBiE (arXiv:2604.06798): "quantization perturbations not only distort expert outputs but also propagate to the gating distribution, inducing expert-shift"
- VSRAQ (arXiv:2606.05688): "rerouting does not merely introduce a continuous numerical error; it sends tokens through different experts and therefore changes the computation path itself"

### 5.5 Lean 4 Formalization Suggestion

```lean4
/-- MoE approximation error bound (Dar 2025) -/
theorem moe_approximation_error_bound
  (d : ℕ) (m : ℕ)
  (beta : ℝ^d → ℝ) (p_x : ℝ^d → ℝ)
  (segments : Fin m → Set ℝ^d)
  (h_partition : isPartition segments)
  (h_smooth : ContDiff ℝ 1 beta) :
  ∃ C : ℝ, approximationError m d beta p_x segments ≤
    sigma_epsilon^2 + C * d / (m : ℝ)^(2/d : ℝ) :=

/-- MoE quantization error propagation -/
theorem moe_quantization_error_propagation
  (W_u W_d : Matrix (Fin d_h) (Fin d) ℝ)
  (epsilon_u epsilon_d : ℝ)
  (h_quant : quantizationError W_u ≤ epsilon_u ∧ quantizationError W_d ≤ epsilon_d) :
  ∀ x : ℝ^d, ‖quantizedMoEOutput x - fullPrecisionOutput x‖^2 ≤
    conditionNumber W_d * epsilon_d^2 * ‖x‖^2 + ‖W_d‖^2 * epsilon_u^2 * ‖x‖^2 :=
```

---

## 6. SUMMARY TABLE

| Theorem | Rate | Conditions | Reference |
|---------|------|------------|-----------|
| Zador (Euclidean) | n^{-1/d} error, n^{-2/d} distortion | (r+delta)-moment finite, density h | Graf-Luschgy 2000, Thm 6.2 |
| Zador (S^{d-1}) | n^{-1/(d-1)} error, n^{-2/(d-1)} distortion | Density on sphere | Graf-Luschgy 2000 + manifold reduction |
| 1-D sphere (great circle) | V_n = L^2/(12n^2) exactly | Uniform on geodesic | Roychowdhury 2025 |
| Greedy quantization | O(n^{-1/d}) | L^p moment, control condition | Luschgy-Pages 2015 |
| MoE approximation | O(d/m^{2/d}) | Smooth beta, p_x | Dar 2025 |
| Sphere covering | N ~ (1/eps)^{d-1} | - | Rogers 1963, Dumer 2006 |
| Additive quantization | No guarantee | NP-hard encoding | Babenko-Lempitsky 2014 |

---

## 7. KEY REFERENCES

1. **Graf, S. & Luschgy, H.** (2000). "Foundations of Quantization for Probability Distributions". Lecture Notes in Mathematics 1730, Springer.
2. **Zador, P.L.** (1982). "Topics in the asymptotic quantization of continuous random variables". IEEE Trans. Inf. Theory, 28(2), 139-149. (Originally 1966 PhD thesis)
3. **Luschgy, H. & Pages, G.** (2015). "Greedy vector quantization". J. Approx. Theory, 198, 111-131.
4. **Babenko, A. & Lempitsky, V.** (2014). "Additive Quantization for Extreme Vector Compression". CVPR 2014.
5. **Dar, Y.** (2025). "Mixture of Many Zero-Compute Experts: A High-Rate Quantization Theory Perspective". arXiv:2510.03151.
6. **Aydin, A.D.** (2025). "Asymptotics of the quantization problem on metric measure spaces". arXiv:2503.18779.
7. **Roychowdhury, M.K.** (2025). "Optimal Quantization on Spherical Surfaces". arXiv:2511.05099.
8. **Rogers, C.A.** (1963). "Covering a sphere with spheres". Mathematika, 10, 157-164.
9. **Dumer, I.** (2006). "Covering spheres with spheres". arXiv:math/0606002.
10. **Nemhauser, G., Wolsey, L., Fisher, M.** (1978). "An analysis of approximations for maximizing submodular set functions". Math. Programming, 14, 265-294.
11. **Das, A. & Kempe, D.** (2011). "Submodular meets spectral: greedy algorithms for subset selection". ICML 2011.
12. **Boutoille, G. & Pages, G.** (2026). "Zador Theorem for optimal quantization with respect to Bregman divergences". arXiv:2604.02354.

