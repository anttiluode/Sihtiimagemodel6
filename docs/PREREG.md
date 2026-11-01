# Sihtiimagemodel6 — preregistration

Written and committed **before** any test observation is generated. Parameters are chosen by
`tune.py` on validation crops only and committed in `params.json` together with this file.

## Question

Model 2's detail stage failed two ways: predicting the *average* detail was too smooth (0.115× the
real fine-band energy), sampling detail independently was grainy (1.59×). Model 4 found the states
that help natural crops are local edge states. The Nov 1 conversation (Antti, Astra, Claude) proposed
that the missing ingredient is **coordination**: local detail atoms that *compete for one shared
reconstruction*, so that neighbouring choices have to agree.

Model 6 tests that on the cleanest version of the detail problem: **see back through Sihti's own
fog.** The observation is the Sihti core at depth 8,
`y = purify^8(x) + N(0, 0.01²)` (σ ≈ 2 px of heat diffusion, Neumann borders), and every method must
put back the residue bands the sieve removed. 32×32 grey crops, CPU only.

## Data (vendored unchanged from Model 4)

Train/validation: crops of 15 scikit-image photographs (train = top 75 %, validation = bottom 25 %).
Test: 64 crops of three *different* photographs (`camera`, `coffee`, `clock`), observation noise seed 23.
Temporal test: 24 integer camera pans (8 per test photograph, 6 frames, |vx| ≤ 2, |vy| ≤ 1),
seed 31, every frame observed through the same fog with fresh noise and reconstructed **independently**.

## Arms (all receive only y; all learned parts fit on training crops)

| Arm | What |
|---|---|
| `wiener` | exact LMMSE under a full-image Gaussian (Ledoit-Wolf). The optimal linear focus. |
| `wiener_sample` | a sample of the same Gaussian posterior instead of its mean (Model 2's "sampled residue") |
| `independent` | 64 learned 8×8 atoms; every overlapping patch sparse-codes its own fogged pixels alone, decisions averaged |
| `epll` | Model 4's engine on this fog: 20-state patch GMM, hard state per patch, EPLL half-quadratic schedule |
| `csc` | the **same 64 atoms** at every position competing for one shared reconstruction under the true fog (convolutional sparse coding, FISTA; LCA's objective) plus a smooth core whose detail is penalised |

## Metrics (Astra's correction adopted)

Fine-band energy near 1.0× does **not** establish correct detail. Scored: pixel MSE; edge **position**
(Canny F-score, 1.5 px tolerance, fixed absolute thresholds); edge **orientation** (structure-tensor
angle error at truth edges); temporal **flicker** (mean temporal std over world pixels seen ≥ 4 px
inside every frame; the truth's is 0). Fine-band energy ratio (Sihti r0+r1) is reported only.

Paired bootstrap over test images (2000 resamples, seed 0) gives 95 % CIs of MSE ratios.

## Gates

| Gate | Claim | Pass condition |
|---|---|---|
| **G1** | focus beyond linear | MSE(csc)/MSE(wiener) ≤ 0.95, CI upper < 1, **and** edge-F(csc) > edge-F(wiener) |
| **G2** | coordination is what pays (same atoms, same sparsity) | MSE(csc)/MSE(independent) ≤ 0.95, CI upper < 1, **and** orientation error(csc) < independent |
| **G3** | beats Model 4's state engine on the fog | MSE(csc)/MSE(epll) ≤ 0.97, CI upper < 1 |
| **G4** | sharpness without temporal instability | flicker(csc) ≤ 1.25 × flicker(wiener) **and** flicker(csc) ≤ flicker(independent) |
| **G5** | *instrument check, declared in advance:* the right energy in the wrong places | band-ratio(wiener_sample) ≥ 0.9 **and** MSE(wiener_sample) > MSE(wiener) **and** edge-F(wiener_sample) < edge-F(csc) |

Report-only (no gate): the same arms and parameters at fog depth 16 (σ ≈ 2.8 px).

## Pilot disclosure

Before writing this file, a pilot on 16 **validation** crops (no test data) gave MSE
wiener 0.00217, epll 0.00204, independent 0.00201, csc 0.00182, wiener_sample 0.00470
(band ratio 1.54). That pilot shaped the tuning grids in `tune.py`; it is why these gates exist
in this form.

A smoke test of the figure code (before this file was committed) ran 4 validation crops and **one**
pan over the test photograph `camera` (seed 5, not the preregistered seed 31; 60 CSC iterations,
untuned parameters). Its numbers were seen: flicker wiener 0.027, independent 0.019, csc 0.014.
Disclosed because that photograph is in the temporal test.

---

## Amendment 1 — before any test run (Nov 1, after Astra's review)

Committed before `run6.py` has ever been run on test data. Nothing below was chosen by looking at
test results; the only test-photograph exposure is the smoke test disclosed above.

**1. The fog is not a blur — description corrected.** Away from borders, `purify` at dt = 0.25
replaces a pixel by the mean of its four neighbours, so one step multiplies the mode (kx, ky) by
(cos kx + cos ky)/2. Eight steps kill the middle frequencies but **pass the checkerboard corner**:
the highest 32×32 Neumann mode keeps 0.995⁸ ≈ 96 % of its amplitude (sign-flipping each step).
"σ ≈ 2 px of heat diffusion" above is wrong; it is a band-stop. (Found by Astra; verified, and
now a test.) The primary experiment keeps the preregistered operator because it is the one the
whole Sihti line uses — but its results say nothing about ordinary blur on their own.

**2. Second fog, gated: true heat diffusion with matched time.** The same Laplacian integrated
over the same time (8 × 0.25 = 2) in 32 steps of 1/16: a real low-pass (all eigenvalues in
[0, 1]; the highest mode falls to ~10⁻¹⁰). Every arm is re-tuned for it on validation
(`tune_amend.py`). Gates **G1, G2, G2b, G3 are repeated on the heat fog** with the same thresholds.
A claim about seeing through fog in general needs both.

**3. Tighter coordination control (Astra).** `independent` differs from `csc` in more than
coordination (local operator, no smooth core). New arm `no_inhibition`: the **same atoms, same
full operator, same smooth core** as `csc`; every atom fits the core's residual **alone**
(LCA with its lateral-inhibition term deleted: u = b, a = scale·soft(u, λ)/‖ASe_j‖²), λ and scale
tuned on validation. New gate:

| Gate | Claim | Pass condition |
|---|---|---|
| **G2b** | competition itself is what pays | MSE(csc)/MSE(no_inhibition) ≤ 0.95, CI upper < 1, **and** edge-F(csc) > edge-F(no_inhibition) |

G2 is kept and renamed "beats independent patches": it compares two whole methods, not coordination alone.

**4. Tuning history.** Stage 1 (`tune.py`) put three choices on grid edges and showed CSC was not
converged at 400 iterations (800 iterations scored *worse* on validation, i.e. early stopping was
acting as a regulariser). Stage 2 (`tune_extend.py`) extended those grids and made the iteration
count an explicit choice between 400 and 1600, recording how far 400 iterations is from the
objective's optimum. Every value and every grid edge is logged in `params.json`.

**5. Report-only:** the Sihti fog at depth 16 (Sihti parameters). The temporal test (G4) stays
on the primary fog; frames are still reconstructed independently — it tests consistency, not
memory.

**6. Early stopping, disclosed.** Stage 2 found that running CSC near convergence (1600 iterations)
is *worse* on validation (MSE 0.00254) than the chosen 400 iterations (0.00231), although 400
iterations is only 1.9 % above the objective's optimum. Part of CSC's advantage is therefore early
stopping acting as an extra regulariser, not the objective alone. The gated arm stays the validated
400-iteration solver; a report-only arm `csc_converged` (1600 iterations, same λ, γ) is added to the
primary test so the size of that dependence is visible. EPLL's β scale was still on its grid edge
after stage 2 and is extended once more in stage 3.
