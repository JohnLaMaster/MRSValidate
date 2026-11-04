# Manuscript updates arising from the `mrsvalidate` implementation

This file tracks every change the manuscript (*Laying the Foundation for the
Clinical Translation of Deep Learning-based Spectral Fitting Models in MRS*)
needs because of work on this repository: errors found while implementing the
equations, places where the library now makes a choice the manuscript leaves
open, and library additions that need new text.

- Locations use equation labels and section titles. Line numbers refer to
  `assets/dev/Laying_the_Foundations_compiled.tex` as of 2026-10-10 and will
  drift as the Overleaf version changes.
- Notation follows the manuscript: $x$ = DL model output, $y$ = reference.
- Status: **[ ]** open, **[x]** done in the manuscript, **[?]** waiting on a
  decision (see [Open decisions](#open-decisions)).
- Bibliography entries marked *(verify)* were written from memory and must be
  checked against the publisher record before use.

---

## 1. Corrections to existing equations and text

### 1.1 [x] MeAPE: stray $100/N$ factor — `eqn:MeAPE`
Done (2026-10-07). Also check that the bias-corrected line has
$y_n$ (not $y$) in the denominator.

### 1.2 [x] Bias-corrected MAPE / MeAPE — `eqn:MAPE w/correction`, `eqn:MeAPE`
Done (2026-10-10). The signed relative error must be centered *before* the
absolute value is taken:
$\frac{100}{N}\sum_n \left|\frac{x_n-y_n}{y_n} - \frac{MPE}{100}\right|$ and
$100\cdot\mathrm{med}\left|\frac{x_n-y_n}{y_n} - \frac{MePE}{100}\right|$.
Library: `errors.bias_corrected_mape`, `errors.meape(definition="bias_corrected")`.

### 1.3 [x] Coefficient of determination: missing square — `eqn:r2` (~l. 434)
```latex
\\ \textnormal{ and } SS_{tot} &= \sum_i (y_i - \Bar{y})^2
```

### 1.4 [x] Power SNR — `eqn:power snr` (~l. 69)
Agreed so far: the second line must square the magnitude (it currently sums
$|y|$, i.e. mean magnitude, not power):
```latex
&= \sqrt{\frac{1}{\sigma_{noise}^2}\cdot \frac{1}{Ns}\sum_{n=1}^{Ns}\left(\sqrt{\Re(y)^2 + \Im(y)^2}\right)^2}
```
Checked 2026-11-02 against the 2026-10-30 flattened version: magnitude is now
squared and the index clash is resolved (metabolite index changed to $m$,
samples to $n$). Still open: exponent sign, and $\mathcal{I}$ (where-text) /
$\mathcal{R}(x)$ (spectral SNR where-text).

Original notes:
- **Index clash.** $n$ labels both the metabolite ($mSNR_P(n)$, $A_n$, $x_n$,
  $d_n$) and the summation over samples. Recommended fix (keeps all existing
  metabolite notation, treats $t$ as the fixed time vector): replace the sum by
  a vector norm,
  ```latex
  \textit{mSNR}&\textsubscript{\textit{P}}(n) = \\&\sqrt{\frac{1}{\sigma_{noise}^2}\cdot \frac{1}{Ns}\left\| y \doteq A_n x_n(t) e^{-t(d_n-gt)} \right\|_2^2}\\
   &= \sqrt{\frac{1}{\sigma_{noise}^2}\cdot \frac{1}{Ns}\left\| \sqrt{\Re(y)^2 + \Im(y)^2}\, \right\|_2^2}
  ```
  and add to the "where" text: "$\|\cdot\|_2$ denotes the Euclidean norm over
  the $Ns$ samples, with operations on $t$ applied element-wise".
- **Exponent sign.** $e^{-t(d_n-gt)} = e^{-d_n t + g t^2}$ grows with time; a
  Voigt decay is $e^{-t(d_n+gt)}$, unless $g$ is defined as negative elsewhere.
- **Symbols in the "where" text.** $\mathcal{I}$ → $\Im$ (power SNR) and
  $\mathcal{R}(x)$ → $\Re(x)$ (spectral SNR, `eqn:spectral snr`), to match the
  equations.

### 1.5 [x] Deming slope written for the reverse regression — `eqn:deming_slope` (~l. 1631)
The objective (`eqn:deming_objective`), the intercept and `eqn:deming_lambda`
all regress $x$ (DL) on the latent value measured by $y$ (reference). The slope
equation is the one for $y$ on $x$; taken literally it returns the reciprocal
slope (0.833 instead of 1.2 on noiseless data). With $\lambda=\sigma_y^2/\sigma_x^2$:
```latex
\begin{equation}\label{eqn:deming_slope}
    \hat{\beta}_1 =
    \frac{
        \lambda\, S_{xx} - S_{yy}
        + \sqrt{\left(\lambda\, S_{xx} - S_{yy}\right)^2
                + 4\,\lambda\, S_{xy}^{\,2}}
    }{
        2\,\lambda\, S_{xy}
    }
\end{equation}
```
Library: `method_comparison.deming` (already uses this form).

### 1.6 [x] Contradictory definitions of the error-variance ratio $\lambda$
Checked 2026-11-02: none of the three locations below is changed yet
(Deming text, Silveira table and $\lambda_{i,n}$ all still use the old ratio).
`eqn:deming_lambda` defines $\lambda=\sigma_y^2/\sigma_x^2$ (reference/DL), and
the limiting cases in the Deming text ($\lambda=1$ orthogonal, $\lambda\to0$
OLS of $x$ on $y$) are correct for that definition. Make the following agree with it:
- Deming text after `eqn:deming_objective` (~l. 1630): "$\lambda = \sigma_x^2/\sigma_y^2$" → $\sigma_y^2/\sigma_x^2$.
- Silveira table, Test 2 structural $H_0$: $\lambda=\mathrm{V}[\epsilon]/\mathrm{V}[\delta]$ → $\mathrm{V}[\delta]/\mathrm{V}[\epsilon]$ (the Test 2 text, ~l. 1588, already has $\mathrm{V}[\delta]/\mathrm{V}[\epsilon]$; $\delta$ is the error of $y$).
- Mixed Test 3 text after `eqn:me_test3` (~l. 1736): $\lambda_{i,n}=\sigma^2_{\mathrm{DL}}/\sigma^2_{\mathrm{ref}}$ → $\sigma^2_{\mathrm{ref},i,n}/\sigma^2_{\mathrm{DL},i,n}$.

### 1.7 [x] Silveira Test 1 text swaps $x$ and $y$ (~l. 1586)
"regressing the pairwise differences $y_i - x_i$ on the mean-centered
reference values $x_i - \bar{x}$" → "regressing $x_i - y_i$ on the
mean-centered reference values $y_i - \bar{y}$", matching the table and
`eqn:me_test1`. Correspondingly, "the regression intercept $\alpha$
corresponds directly to the mean difference $\bar{x} - \bar{y}$".

### 1.8 [x] Translation correction cannot change the Test 2 result (~l. 1588, ~l. 1726)
The Test 2 slope is an OLS slope of $x_i-y_i$ on $x_i+y_i$. Subtracting a
constant $c$ from $x$ shifts both the response and the regressor by constants,
so $\hat\beta$, its standard error and its $p$-value are unchanged (only the
intercept moves, by $-c(1-\hat\beta)$). Verified numerically in the library.
For Test 3 the Deming slope is likewise invariant; only the intercept shifts.
Suggested replacement for the Test 2 sentence:
```latex
Because the Test~2 slope is unaffected by a constant shift of either method,
a non-zero bias established by Test~1 does not alter the Test~2 null; the
translation correction of Silveira et al.~\cite{Silveira2024} affects only the
intercept and is therefore relevant to Test~3.
```
and for the mixed-effects Test 2 (~l. 1726), delete "As in the per-metabolite
case, a translation correction ... prior to evaluating the Test~2 null." or
reword the same way. The same Test 2 paragraph also needs 1.17 and 1.18.

### 1.9 [x] Typo in `eqn:me_test3` (~l. 1733)
`\epsilon{i,n}` → `\epsilon_{i,n}`.

### 1.10 [x] Cross-classified design effect pairs each ICC with the wrong cluster size — `eqn:deff_cc`
Each ICC must be multiplied by (cluster size − 1), where the cluster size is
the number of *observations* sharing that random effect: $S\bar N$ per
metabolite and $K\bar N$ per subject, not $\bar K$ and $\bar N$. In the
library's simulation the current formula gave a design effect of 5.4 against a
true value of about 50, overstating $N_{\mathrm{eff}}$ roughly ten-fold.
```latex
\begin{equation}\label{eqn:deff_cc}
    \textnormal{deff}_{\textnormal{CC}} =
        1 + \rho_i\,(\tilde{m}_i - 1) + \rho_n\,(\tilde{m}_n - 1),
    \qquad
    \tilde{m}_i = \frac{\sum_{i=1}^{K} N_i^2}{N_{tot}},
    \quad
    \tilde{m}_n = \frac{\sum_{s=1}^{S} N_s^2}{N_{tot}}
\end{equation}
\noindent where $\rho_i$ is the ICC associated with the metabolite clustering dimension,
$\rho_n$ is the ICC associated with the subject clustering dimension, $N_i$ and $N_s$ are
the numbers of observations for metabolite $i$ and subject $s$, and $N_{tot}$ is the total
number of observations. $\tilde{m}_i$ and $\tilde{m}_n$ are the size-weighted mean cluster
sizes of the two dimensions; in a balanced design with $\bar{N}$ spectra per subject they
reduce to $\tilde{m}_i = S\,\bar{N}$ and $\tilde{m}_n = K\,\bar{N}$.
```
`eqn:neff_cc` can stay ($K S \bar N = N_{tot}$ when every spectrum contains all
metabolites); writing it as $N_{tot}/\mathrm{deff}_{CC}$ also covers
unbalanced designs. Library: `clustering.cross_classified_icc`.

### 1.11 [x] Robust limits of agreement cover ~81 %, not 95 % — Bland–Altman, "Robust Estimation of Bias" (~l. 1517)
*Revised 2026-11-02.* The earlier version of this item (multiply
$\widetilde{\sigma}_d$ by 1.4826) was applied and checked off; it is replaced by
the approach below, so the sentence added for the earlier version should be
removed or replaced.

**Problem.** The "Robust Estimation of Bias" paragraph estimates the spread of
the differences with $\widetilde{\sigma}_d$ "defined in
Section~\ref{sec:Eval:subsec:error:subsubsection:dispersion}", i.e.
`eqn:np variance`, $\widetilde\sigma=\sqrt{\mathrm{med}((\widetilde\mu-x_n)^2)}$.
That quantity is (essentially) the unscaled median absolute deviation, which
is $0.6745\,\sigma$ for normal data, so $\widetilde d \pm 1.96\,\widetilde\sigma_d$
covers only about 81 % of differences
($2\Phi(1.96 \times 0.6745) - 1 \approx 0.81$).

**Decision.** Keep `eqn:np variance` unchanged (it is an intentional
definition used elsewhere, e.g. `eqn:np cv`). In the Bland–Altman paragraph,
stop pointing to it and define the robust SD of the differences directly as
the normal-consistent MAD, which is what the library and standard software
(R `mad()`, `scipy.stats.median_abs_deviation(scale="normal")`) compute. Use a
new symbol, $\widetilde{s}_d$, so it is not confused with $\widetilde{\sigma}$.

**What to change in the manuscript.**
1. In "Robust Estimation of Bias" (flattened 2026-10-30 version, l. 3870),
   replace the sentence
   ```latex
   The dispersion of differences is correspondingly estimated using the nonparametric standard deviation $\widetilde{\sigma}_d$ defined in Section~\ref{sec:Eval:subsec:error:subsubsection:dispersion}, and the IQR of the differences is reported as a supplementary robust characterization of spread.
   ```
   (and the earlier 1.11 sentence, if already inserted after it) with
   ```latex
   The dispersion of differences is correspondingly estimated using the normal-consistent median absolute deviation, $\widetilde{s}_d = 1.4826\,\mathrm{med}\,\lvert d_n - \widetilde{d}\rvert$, where the constant $1/\Phi^{-1}(0.75) \approx 1.4826$ makes $\widetilde{s}_d$ a consistent estimator of the standard deviation under normality~\cite{Rousseeuw1993}, so that the robust limits of agreement $\widetilde{d} \pm 1.96\,\widetilde{s}_d$ cover 95\,\% of differences; without this constant the coverage is approximately 81\,\%. Where no distributional form is assumed, the limits may instead be taken as the empirical 2.5\,\% and 97.5\,\% quantiles of $d_n$. The IQR of the differences is reported as a supplementary robust characterization of spread.
   ```
2. Search the Bland–Altman and Silveira subsections for any other
   $\widetilde{\sigma}_d$ and change it to $\widetilde{s}_d$ (none found in
   the 2026-10-30 flattened file besides the sentence above, but check the
   current Overleaf version).
3. Add `Rousseeuw1993` to the bibliography (entry in section 4).
4. No change to `eqn:np variance`.

Related: the 2.2 text ("the absolute-residual model is multiplied by
$1.4826$") applies the same constant to the regression-based robust limits
and is consistent with this.

**Library.** No computational change: `bland_altman(scale="robust_sd")`
already computes $1.4826\,\mathrm{med}|d_n-\widetilde d|$
(`method_comparison.py`, `_ba_point_estimates`). Only its docstring needs
correcting: it calls the MAD "the manuscript's unscaled
$\widetilde{\sigma}_d$, cf. `_numerics.robust_sd`", which is true only for odd
$N$; it should instead describe it as the normal-consistent MAD, matching the
manuscript's $\widetilde{s}_d$.
Library: `bland_altman(scale="robust_sd", loa_method="normal"|"quantile")`.

### 1.12 [ ] Passing–Bablok slope estimator — `eqn:pb_slopes`, `eqn:pb_slope` (~l. 1603–1609)
The printed estimator takes the upper-middle order statistic when the number of
slopes is even, keeps $S_{ij}=-1$, and drops all pairs with $y_i=y_j$. The
original estimator (Passing & Bablok 1983), now the library default, differs on
all three. Replace the sentence after `eqn:pb_slopes` and `eqn:pb_slope` with:
```latex
\noindent where $y_i$ and $x_i$ are the reference and \DL model values for the
$i$-th observation. Pairs with $y_i = y_j$ and $x_i = x_j$ are discarded; pairs
with $y_i = y_j$ and $x_i \neq x_j$ are assigned $S_{ij} = +\infty$ or
$-\infty$ according to the sign of $x_j - x_i$; and slopes with $S_{ij} = -1$
are discarded. With $N_S$ the number of retained slopes and $K$ the number
satisfying $S_{ij} < -1$, the slope estimate is the shifted median
\begin{equation}\label{eqn:pb_slope}
    \hat{\beta}_1 =
    \begin{cases}
        S_{\left((N_S+1)/2 + K\right)}, & N_S \text{ odd},\\[2pt]
        \tfrac{1}{2}\left(S_{(N_S/2 + K)} + S_{(N_S/2 + K + 1)}\right), & N_S \text{ even},
    \end{cases}
\end{equation}
```
Library: `passing_bablok(estimator="passing_bablok_1983")`; the printed version
is kept as `estimator="manuscript_literal"`.

#### 1.12 addendum (2026-11-02): regime split — Theil–Sen *in silico*, Deming *in vivo*, Passing–Bablok kept
The replacement text above remains correct for the Passing–Bablok subsection
(index by $N_S$, average the two middle slopes, published rules for tied
reference values and $S_{ij} = -1$). In addition:

**Decision.** *In vivo* (both methods have error) → Deming. *In silico*
(exact reference) → Theil–Sen as the primary robust estimator, with
Passing–Bablok reported alongside for comparability with the literature.
Reason: Passing–Bablok assumes both methods carry error (error-variance ratio
≈ $\beta_1^2$); its shift by $K$ is what accounts for the reference error, so
with an exact reference it biases the slope slightly upward. Theil–Sen is the
same median of pairwise slopes without the shift, i.e. the robust counterpart
of OLS of $x$ on $y$, which the Deming text already names as the exact-reference
($\lambda \to 0$) limit.

**Manuscript changes** (full instructions and the current text in
`assets/dev/HANDOVER_regression_section.md`, edits E1–E5):
- E1: the "Regression-Based Comparisons" intro states the mapping explicitly.
- E2: Passing–Bablok kept; state its assumption; $N_S$ indexing and
  averaging; published special-pair rules; replace the incorrect rationale
  for the shift by $K$ ("ensures consistency when the true slope is
  negative") with the correct one (median of pairwise angles, symmetric in
  the two methods; assumes a positive relationship); note the slight upward
  bias with an exact reference.
- E3: new "Theil–Sen Regression" subsection between Passing–Bablok and
  Deming: drop pairs with $y_i = y_j$, median of the retained slopes, same
  intercept, Sen's interval with tie correction, same CUSUM/reporting.
- E4: Deming states it is the *in vivo* estimator; the *in silico* sentence
  points to Theil–Sen; the Passing–Bablok fallback keeps the condition
  "errors of similar magnitude".
- E5: bibliography `Theil1950`, `Sen1968` (section 4).

**Knock-on items.** 2.3: add the Theil–Sen (Sen) interval. 2.4: the CUSUM
applies to both; for Theil–Sen the wild bootstrap flips residuals vertically
($\lambda = 0$). 3.1: Figure 5 panel C uses Theil–Sen after regeneration, so
the caption numbers change. Section 5: the bias checks (exact reference:
Passing–Bablok 1.017 vs Theil–Sen 0.999 for a true slope of 1) are B-kind
numbers; keep them out of the text or back them with a script.

**Library.** `passing_bablok` (published rules) is unchanged apart from
removing `estimator="manuscript_literal"`; new `theil_sen` for the *in
silico* regime with the same result fields; Figure 5 panel C switches to it.

### 1.13 [x] Inconsistent sample-variance normalisation in the Monte Carlo section — `sections/sec6.1_info-standardizedCharacterization.tex` (l. 245, 249)
$\hat\sigma^2_{\mathrm{DL},m,j}$ uses $1/(K-1)$ while $\widehat{RMSE}_{m,j}$
uses $1/K$, so the empirical decomposition
$\widehat{RMSE}^2 = \widehat{\mathrm{Bias}}^2 + \hat\sigma^2_{\mathrm{DL}}$
holds only up to the factor $(K-1)/K$ on the variance.

**Decision (2026-11-02): keep $1/(K-1)$ and add a disclaimer.** Switching to
$1/K$ would make the identity exact, but the $1/K$ variance is biased low by
$(K-1)/K$. $\hat\sigma_{\mathrm{DL}}$ is used mainly against the CRLB
($R_{m,j}$, `eqn:variability ratio`; $\Delta V_{m,j}$; the sub-CRLB
analysis), and a low-biased variance pushes all of them toward apparent
sub-CRLB precision, which is the effect that analysis is meant to detect. The
discrepancy ($1/K$ of the variance) is far smaller than the sampling
uncertainty of $\hat\sigma^2_{\mathrm{DL}}$ itself (relative standard error
$\approx\sqrt{2/(K-1)}$ for approximately normal predictions): 1 % vs 14 % at
$K = 100$, 5 % vs 32 % at $K = 20$.

**Change.** Insert directly after the $\widehat{RMSE}_{m,j}$ equation (l. 249,
before "The additional index $j$ is essential …"):
```latex
\noindent The two estimators use different normalizations: $\hat{\sigma}_{\DL,m,j}^{2}$ is the unbiased sample variance with the factor $1/(K-1)$, whereas $\widehat{RMSE}_{m,j}$ averages over all $K$ realizations. The empirical decomposition therefore holds exactly in the form
\begin{equation}
    \widehat{RMSE}_{m,j}^{2} = \widehat{\textnormal{Bias}}_{m,j}^{2} + \frac{K-1}{K}\,\hat{\sigma}_{\DL,m,j}^{2},
\end{equation}
and only approximately with $\hat{\sigma}_{\DL,m,j}^{2}$ itself. The unbiased form is retained because $\hat{\sigma}_{\DL,m,j}$ is compared with the \CRLB reference (Section~\ref{sec:Analy&Visual:subsec:infoStand:subsubsec:subcrlb var}): normalizing by $K$ would underestimate the variance by the factor $(K-1)/K$ and bias $R_{m,j}$ and $\Delta V_{m,j}$ toward apparent sub-\CRLB precision. The difference amounts to a fraction $1/K$ of the variance, which is small compared with the sampling uncertainty of $\hat{\sigma}_{\DL,m,j}^{2}$ itself, approximately $\sqrt{2/(K-1)}$ in relative terms for approximately normally distributed predictions (e.g., $1\,\%$ against $14\,\%$ for $K = 100$), and it is therefore negligible for the numbers of realizations used in \MC experiments.
```
Library: `monte_carlo.monte_carlo_condition` already uses the same convention
(`ddof=1` variance, $1/K$ RMSE); no change needed.

### 1.14 [x] Opposite error sign convention in the multivariate section (~l. 918)
$\mathbf{e} = \mathbf{c} - \hat{\mathbf{c}}$ is reference minus prediction,
the reverse of the $x - y$ convention used elsewhere. Harmless for $D^2$, but
for consistency use $\mathbf{e} = \hat{\mathbf{c}} - \mathbf{c}$.

### 1.15 [x] Broken cross-reference (~l. 1327, ~l. 1523)
`\ref{subsubsec:Lin's rho}` → `\ref{sec:Eval:subsec:RegMetrics:subsubsec:Lin's rho}`.

### 1.16 [x] ECWAS: max-type moments are not analytic (~l. 612)
"derived analytically from the null distributions as functions of sample size"
is true for $WI$, $AB$, $AS$, but not for $\max(AS,S)$ in `eqn:funato`. Covered
by the new paragraph in 2.1.

### 1.17 [x] Silveira Test 2 text swaps $x$ and $y$ (Equivalence Testing, Test 2 paragraph)
Same swap as 1.7: "the regression of $y_i - x_i$ on $x_i + y_i$" →
"the regression of $x_i - y_i$ on $x_i + y_i$", matching the table and
`eqn:me_test2`. Consistency only: flipping the sign of the response flips the
sign of the slope, so the test of a zero slope is unchanged.

### 1.18 [x] Shukla's Test 2 equivalence holds only without proportional bias (Equivalence Testing, Test 2 paragraph)
$\mathrm{Cov}(x-y,\,x+y) = \mathrm{V}[x] - \mathrm{V}[y]$. With
$x = \alpha + \beta\xi + \epsilon$ and $y = \xi + \delta$ this is
$(\beta^2 - 1)\,\mathrm{V}[\xi] + \mathrm{V}[\epsilon] - \mathrm{V}[\delta]$,
so "the slope is exactly zero if and only if $\lambda = 1$" holds only when
$\beta = 1$. A constant offset is harmless, but proportional bias makes Test 2
reject even when the precisions are equal (or mask unequal precisions). First
check whether Silveira et al.'s structural model fixes the slope at 1, in which
case "under the specified structural measurement-error model" already covers
it. If not, add after the sentence ending "$\lambda = \text{V}[\delta] /
\text{V}[\epsilon] = 1$":
```latex
This equivalence requires that the two methods share a unit slope; under
proportional bias, $\mathrm{Cov}(x - y,\, x + y)$ also contains the term
$(\beta^2 - 1)\,\mathrm{V}[\xi]$, so Test~2 is interpretable as a test of
precision only in the absence of proportional bias.
```
Don't point to the Test 1 slope as the check: it estimates
$\left((\beta - 1)\,\mathrm{V}[\xi] - \mathrm{V}[\delta]\right)/\mathrm{V}[y]$, which is pulled negative by
reference error *in vivo* (exact only *in silico*, where $\delta = 0$).
The same caveat applies to the mixed-effects Test 2 (`eqn:me_test2`).

### 1.19 [x] Mixed-effects Test 2 inherits the 1.18 caveat (Mixed-Effects Equivalence Testing, Test 2 paragraph after `eqn:me_test2`)
Within metabolite $i$, $\mathrm{Cov}(x_i - y_i,\, x_i + y_i) =
(b_i^2 - 1)\,\mathrm{V}[\xi_i] + \mathrm{V}[\epsilon_i] - \mathrm{V}[\delta_i]$,
with $b_i$ the proportional bias of metabolite $i$. Hence (a) the fixed slope
$\beta$ is zero iff $\lambda = 1$ only if every $b_i = 1$; (b) $\tau^2_\beta$
also absorbs inter-metabolite heterogeneity in proportional bias, not only in
the precision ratio; (c) the sentence that a Test 1 bias "does not alter the
Test 2 null" is true for constant bias only. Same precondition as 1.18: skip
if Silveira et al.'s structural model already fixes the slope at 1. As in
1.18, don't point to the Test 1 slope (`eqn:me_test1`) as the check.
Edits refer to the flattened text; the repo copy of
`sections/sec8_moodelComparison.tex` (l. ~258–265) still has the older
"translation correction … is applied" sentence in place of edit 3's target.

1. Replace "By the Shukla result~\cite{Shukla1973} underlying Test~2, the fixed
   slope $\beta$ is zero if and only if the model-level error variance ratio
   $\lambda = 1$, and the null $H_{0,2}: \beta = 0$ is the model-level
   precision test." with:
   ```latex
   By the Shukla result~\cite{Shukla1973} underlying Test~2, and provided that, as in the per-metabolite case, the two methods share a unit slope for every metabolite, the fixed slope $\beta$ is zero if and only if the model-level error variance ratio $\lambda = 1$, and the null $H_{0,2}: \beta = 0$ is the model-level precision test. Proportional bias in a metabolite shifts that metabolite's slope away from zero even when the two methods are equally precise (Section~\ref{subsubsec:silveira}).
   ```
2. Replace "The random slope variance $\tau^2_\beta$ quantifies inter-metabolite
   heterogeneity in the precision ratio." with:
   ```latex
   The random slope variance $\tau^2_\beta$ quantifies inter-metabolite heterogeneity in the precision ratio, together with any inter-metabolite heterogeneity in proportional bias.
   ```
3. Replace "Because the Test~2 slope is unaffected by a constant shift of either
   method, a non-zero bias established by Test~1 does not alter the Test~2
   null." with:
   ```latex
   Because the Test~2 slope is unaffected by a constant shift of either method, a constant bias established by Test~1 does not alter the Test~2 null; proportional bias does, as noted above.
   ```

### 1.20 [x] Silveira Test 3 paragraph: $\lambda$ is not estimated in Test 2, and Test 2 needs no translation correction (Equivalence Testing, Test 3 paragraph, 10/30 draft l. 4202–4204)
Current text (10/30 draft), with the two problem phrases in bold:
> Test 3 (Agreement) assesses concordance with the structural bisector line
> using Deming regression \cite{Linnet1998}, which accounts for measurement
> error in both variables by incorporating the error variance ratio $\lambda$
> **estimated in Test~2**. […] Analytical intervals for the Deming intercept
> and slope are obtained from jackknife standard errors […].
> **As with Test~2**, a translation correction for residual bias is applied
> when Test~1 is not satisfied. Section~\ref{…regression} describes […].

1. **"$\lambda$ estimated in Test~2".** Test 2 tests $\lambda = 1$; it does not
   estimate $\lambda$, and $\lambda$ is not identifiable from single paired
   measurements (D2). It is supplied from independent precision estimates, as
   the mixed-effects Test 3 text already says for $\lambda_{i,n}$. Replace the
   first sentence with:
   ```latex
   \textcolor{black}{\textbf{Test~3 (Agreement)}} assesses concordance with the structural bisector line using Deming regression \cite{Linnet1998}, which accounts for measurement error in both variables through the error variance ratio $\lambda$ (Eqn.~\ref{eqn:deming_lambda}). Because $\lambda$ cannot be estimated from single paired measurements, it is supplied from independent precision estimates of the two methods; Test~2 only tests whether $\lambda = 1$. In the \textit{in silico} regime the reference is exact, so $\lambda = 0$ and Deming regression reduces to ordinary least-squares regression of $x_i$ on $y_i$.
   ```
   The last sentence is optional; it makes explicit what the regime pointer at
   the end of the paragraph implies. If a label is added to the Deming
   subsubsection (handover E4, `sec:Reg:subsec:comparison:subsubsec:deming`),
   "(Section~\ref{…deming})" can follow "independent precision estimates of
   the two methods".
2. **"As with Test~2".** The Test 2 paragraph now says a constant bias does
   not alter the Test 2 null, so no translation correction is applied there.
   Replace with:
   ```latex
   A translation correction for residual bias is applied when Test~1 is not satisfied.
   ```

Library follow-up (only if the optional *in silico* sentence is kept):
`method_comparison.deming` and `mixed_effects` Test 3 reject a zero
reference error variance, so the $\lambda = 0$ (OLS) limit cannot currently be
run through them.

---

## 2. New text needed (library behavior the manuscript does not yet describe)

### 2.1 [x] ECWAS critical values — new paragraph after `eqn:funato` (Distributional Similarity, "Hypothesis Tests for DL-Based Studies")
The three components are correlated under $H_0$ (Wilcoxon–AntiSavage
$\approx -0.87 \to -\sqrt3/2$; Ansari-Bradley–AntiSavage $\approx -0.34$), so
$\chi^2_3$ is anti-conservative (7.4 % type-I error at nominal 5 % for
100 + 100 observations; permutation null 4.6 %). Library default:
joint permutation null, 10 000 relabelings.
```latex
\paragraph{Critical values for the composite statistics.}
The three components of Eqns.~\ref{eqn:ecwas} and \ref{eqn:funato} are linear rank
statistics computed from the same pooled ranks and are therefore not independent under the
null hypothesis. For linear rank statistics, the null correlation between two components
equals the correlation between their score functions~\cite{Hajek1999}, so it depends only on
the sample sizes. The Wilcoxon and Ansari-Bradley components are uncorrelated, because the
Ansari-Bradley scores are symmetric about the middle rank. In contrast, the Wilcoxon and
AntiSavage components have a correlation of approximately $-0.87$, which tends to
$-\sqrt{3}/2$ in large samples, and the Ansari-Bradley and AntiSavage components have a
correlation of approximately $-0.34$. Consequently, $T_{ECWAS}$ does not follow a $\chi^2$
distribution with three degrees of freedom, and using that reference distribution inflates the
type-I error rate. K\"{o}ssler and Mukherjee~\cite{Kossler2019} derived the asymptotic null
distributions of the Euclidean and Mahalanobis forms of the statistic, but no closed form is
available for the max-type shape component of Eqn.~\ref{eqn:funato}.

This work therefore obtains critical values from the permutation distribution of the composite
statistic~\cite{Good2005}, following the procedure in
Section~\ref{sec:perf:subsec:bootstrap}. Under the null hypothesis, the group labels of the
pooled sample are exchangeable. The labels are randomly reassigned $P$ times with the group
sizes held fixed. For each relabeling $k$, all three components are recomputed and
standardized by the same null means and standard deviations as the observed statistic,
yielding $T^{(k)}_{\mathrm{perm}}$. Recomputing the components jointly preserves their
correlation structure. The critical value at significance level $\alpha$ is the $(1-\alpha)$
quantile of $\{T^{(k)}_{\mathrm{perm}}\}_{k=1}^{P}$, and the $p$-value follows
Eqn.~\ref{eqn:permutation_pval}, which counts the observed statistic as a member of the
permutation distribution and therefore cannot be zero~\cite{Phipson2010}. For the max-type
variant, $\mu_{\max}$ and $\sigma_{\max}$ are estimated as the mean and standard deviation of
$\max(AS, S)$ over the same relabelings. Because the statistics depend on the data only
through their ranks, the permutation distribution depends only on the two sample sizes when
there are no ties; tied values are assigned averaged (midrank) scores. The procedure is
distribution-free, valid for any sample size, and applies unchanged to both variants. This
work uses $P = 10\,000$ relabelings. The number of relabelings and the random seed should be
reported so that the result can be reproduced exactly.
```
New bibliography entries: `Phipson2010`, `Hajek1999` (see section 4).
Checked 2026-11-02: paragraph is in, but both keys are still missing from
the project's `mybibliography.bib`.

### 2.2 [x] Bland–Altman: how the conditional models and confidence bands are computed — "Regression-Based Limits of Agreement" and "Confidence Intervals and Clinical Acceptability" (~l. 1510–1521)
The manuscript says $\hat\beta(c)$ and $\hat\sigma_d(c)$ are "each modeled by
regression on $c$" and that confidence bands are reported, but not how. Add at
the end of "Regression-Based Limits of Agreement":
```latex
For the parametric estimators, $\hat{\beta}(c)$ is obtained by ordinary least squares
regression of $d_n$ on $c_n$, and $\hat{\sigma}_d(c)$ by least squares regression of the
absolute residuals $|d_n - \hat{\beta}(c_n)|$ on $c_n$, multiplied by $\sqrt{\pi/2}$ so that
it estimates a standard deviation under conditional normality~\cite{Bland1999}. For the
robust estimators (Section~\ref{...robust...}), both regressions use the Theil--Sen
estimator, and the absolute-residual model is multiplied by $1.4826$, since the median
absolute residual of a normal variable is $0.6745\,\sigma$.
```
and replace the first sentence of "Confidence Intervals and Clinical Acceptability" with:
```latex
Confidence intervals are reported on both the estimated bias and the limits of agreement at
each concentration, reflecting the uncertainty arising from finite sample size~\cite{Lu2016}.
They are obtained from a paired bootstrap of the $(x_n, y_n)$ pairs, in which the complete
analysis (bias and scale models) is refitted to each resample and pointwise percentile bands
are taken over a grid of $c$ (Section~\ref{sec:perf:subsec:bootstrap}). For the parametric
constant-variance limits only, the analytical intervals of Bland and Altman~\cite{Bland1999}
may be used instead: $\bar{d} \pm t_{N-1,\,0.975}\,s_d/\sqrt{N}$ for the bias and
$\mathrm{LoA} \pm t_{N-1,\,0.975}\,s_d\sqrt{1/N + 1.96^2/(2(N-1))}$ for each limit.
```
Fill in the `\ref{...robust...}` placeholder with the label of "Robust Estimation of Bias".
Library: `bland_altman(conditional=True, ci_method="bootstrap"|"analytic")`.

### 2.3 [x] Passing–Bablok confidence intervals — "Passing-Bablok Regression" (~l. 1617)
The manuscript states 95 % intervals but not the method. Library default: the
analytic rank-based interval of Passing & Bablok (1983); paired percentile
bootstrap available. Add after `eqn:pb_intercept`:
```latex
Confidence intervals follow Passing and Bablok~\cite{Passing1983}. With
$C_{\gamma} = z_{1-\gamma/2}\sqrt{N(N-1)(2N+5)/18}$, $M_1 = \operatorname{round}\!\left((N_S - C_{\gamma})/2\right)$
and $M_2 = N_S - M_1 + 1$, the $100(1-\gamma)\%$ interval for the slope is
$\left[S_{(M_1+K)},\, S_{(M_2+K)}\right]$, and the interval for the intercept is
$\left[\widetilde{\mu}(x_i - \hat{\beta}_{1,U}\, y_i),\, \widetilde{\mu}(x_i - \hat{\beta}_{1,L}\, y_i)\right]$,
where $\hat{\beta}_{1,L}$ and $\hat{\beta}_{1,U}$ are the lower and upper slope limits.
A paired bootstrap interval (Section~\ref{sec:perf:subsec:bootstrap}) is a distribution-free
alternative when the rank-based approximation is in doubt, for example for small $N$.
```

### 2.4 [ ] Passing–Bablok CUSUM linearity test — "Passing-Bablok Regression" (10/30 draft l. 4254)
The manuscript cites the test but does not define it or its critical value.
D2 is decided (bootstrap default, $\lambda$ supplied), so the earlier split
into a "common part" plus calibration sentences in D2 is replaced by the
single draft below. The asymptotic critical value appears only as the reason
for the bootstrap; the estimator wording is generic so it also covers
Theil–Sen; and it cites only labels that already exist (`eqn:deming_lambda`,
`eqn:permutation_pval`).

**Placement:** end of "Passing-Bablok Regression", after "The
\textit{cumulative sum} linearity test should be reported alongside the
regression results \cite{Passing1983,Bilic-Zulle2011}." and before "A
detected departure from linearity indicates…".
```latex
The test classifies each observation by the sign of its residual $e_i = x_i - (\hat{\beta}_0 + \hat{\beta}_1 y_i)$ and assigns the score $r_i = \sqrt{n_-/n_+}$ if $e_i > 0$, $r_i = -\sqrt{n_+/n_-}$ if $e_i < 0$, and $r_i = 0$ otherwise, where $n_+$ and $n_-$ are the numbers of positive and negative residuals, so that $\sum_i r_i = 0$. The scores are ordered by the position of each observation along the fitted line, and the test statistic is $H = \max_k \bigl|\sum_{i \le k} r_{(i)}\bigr| / \sqrt{n_+ + n_- + 1}$. Because the line is estimated from the same data, the null distribution of $H$ is not the Kolmogorov distribution~\cite{Durbin1973}, and the classical critical value of $1.36$ at $\alpha = 0.05$~\cite{Passing1983} is strongly conservative. Critical values are therefore obtained by a wild bootstrap under the fitted linear model~\cite{Stute1998}: each residual is multiplied by a random sign along a direction set by the error variance ratio $\lambda$ (Eqn.~\ref{eqn:deming_lambda}) and added back to its fitted position, the line is re-estimated with the same estimator, and $H$ is recomputed. The $p$-value follows Eqn.~\ref{eqn:permutation_pval} with the bootstrap replicates in place of permutations.

The bootstrap is calibrated only when this direction matches the error structure of the data. In the \textit{in silico} regime the reference is exact, $\lambda = 0$, and residuals are flipped vertically. In the \textit{in vivo} regime, $\lambda$ cannot be estimated from single paired measurements and is supplied as for Deming regression, from independent precision estimates of the two methods. Overestimating $\lambda$ makes the test conservative and underestimating it makes the test liberal; when $\lambda$ is known only approximately, the test is repeated at $\lambda/2$, $\lambda$, and $2\lambda$, and the largest $p$-value is reported.
```
New bibliography entries: `Durbin1973`, `Stute1998` (neither is in
`mybibliography.bib`).

**Theil–Sen (handover E3):** the draft already defines Theil–Sen in "Theil-Sen
Estimator" (l. 1713, `sec:Eval:subsec:RegMetrics:subsubsec:theil-sen`). The
new Theil–Sen subsection in the regression section should cite that
definition rather than repeat it, and add one sentence: "The linearity test
of Section~\ref{…passing-bablok} applies unchanged, with $\lambda = 0$." Tell
ChatGPT about the existing section when using the handover.

**Library (to do):**
- `method_comparison.passing_bablok`: accept
  `cusum_bootstrap_variance_ratio=0` (vertical flips), so the *in silico*
  sentence above holds for Passing–Bablok as well; it currently requires a
  positive value. `theil_sen` already fixes $\lambda = 0$.
- Optional: a helper or documented recipe for the $\lambda/2$, $\lambda$,
  $2\lambda$ sensitivity check (report the largest $p$-value).
- Simulation supporting the calibration claims (D2 table, section 5 row g):
  add the scratchpad scripts as a reproduction script under
  `assets/publication/`.

### 2.5 [x] Silveira Test 3: analytic intervals — "Equivalence Testing" (~l. 1590–1592)
The manuscript requires analytic $p$-values and bootstrap bands for every test.
For Tests 1–2 the analytic results are the usual $t_{N-2}$ regression
inferences. For Test 3 (Deming) the library uses jackknife standard errors,
which the manuscript does not yet name. Add to the Test 3 description:
```latex
Analytical intervals for the Deming intercept and slope are obtained from jackknife standard
errors~\cite{Linnet1993}, $\widehat{\mathrm{SE}} = \sqrt{\tfrac{N-1}{N}\sum_{j}(\hat{\theta}_{(j)}
- \bar{\theta}_{(\cdot)})^2}$, where $\hat{\theta}_{(j)}$ is the estimate with observation $j$
omitted, and the Bonferroni-adjusted quantile $t_{N-2,\,1-\alpha/4}$.
```
New bibliography entry: `Linnet1993`.

*Status 2026-11-04:* sentence inserted in the 10/30 draft (l. 4203, after the
Bonferroni sentence). Remaining: `Linnet1993` is not yet in
`mybibliography.bib`. The rest of that Test 3 paragraph has two
inconsistencies with recent changes; see 1.20.

### 2.6 [x] Mixed-effects Test 3: estimation procedure — "Mixed-Effects Equivalence Testing", after `eqn:me_test3` (~l. 1736)
The manuscript gives the model but no estimator. Library: two-stage procedure.
Insert after the sentence on $\lambda_{i,n}$:
```latex
Because Eqn.~\ref{eqn:me_test3} combines an errors-in-variables regression with random
effects, it is estimated in two stages. First, for each metabolite $i$, the intercept
$\hat{\alpha}_i$ and slope $\hat{\beta}_i$ of $x_{i,n}$ on the latent reference value, together
with their standard errors, are estimated by York regression~\cite{York2004}, which
generalizes Deming regression to observation-specific error variances
$\sigma^2_{\mathrm{ref},i,n}$ and $\sigma^2_{\mathrm{DL},i,n}$ and reduces to it when these
are constant. Second, the metabolite-level estimates are combined in a random-effects model,
separately for the intercepts and the slopes, with $\tau^2_\alpha$ and $\tau^2_\beta$
estimated by restricted maximum likelihood~\cite{Viechtbauer2005}. Consistent with the
metabolite-balanced aggregation used throughout this work
(Section~\ref{sec:Eval:subsec:multivariate:subsubsec:error}), the fixed effects $\alpha$ and
$\beta$ are the equal-weight means of the metabolite-level estimates, so that metabolites
estimated with higher precision, such as prominent singlets, do not dominate the model-level
comparison. Confidence intervals for $\alpha$, $\beta$, $\tau^2_\alpha$ and
$\tau^2_\beta$ are obtained by a cluster bootstrap that resamples whole spectra (or subjects
\textit{in vivo}) with replacement and repeats both stages for each resample, which preserves
the intra-spectrum correlation between metabolites.
```
The last sentence reflects decision D1 (panel-conditional inference); add
the D1 paragraph below to the same subsection. New bibliography entries: `York2004`, `Viechtbauer2005`.

*Status 2026-11-04:* inserted in the 10/30 draft (l. 4465), followed by the D1
paragraph (l. 4467), before "The random effect variances…". Remaining:
`York2004` and `Viechtbauer2005` are not yet in `mybibliography.bib`.
Library (docstring only): `mixed_effects.py` ~l. 1224 describes
`lambda_{i,n}` as `test_error_variance / reference_error_variance` (DL /
reference), the inverse of the convention fixed in 1.6 and of its own
Notes (~l. 1269); the computation is unaffected.

### 2.7 [x] Small-sample correction for the cluster-robust standard errors — after `eqn:sandwich mean` (~l. 1716)
The manuscript notes that enough clusters are needed (citing Cameron2008) but
gives no correction. Library default: CR2 with Bell–McCaffrey degrees of
freedom below 50 clusters, CR1 with $t(G-1)$ above. In simulation with 10
clusters, the uncorrected estimator with a normal reference rejected a true
null in 11–18 % of cases at nominal 5 %, against 4–5 % with CR2.

**Placement:** in the paragraph after `eqn:sandwich mean`, between "…may be
closer to the number of subjects in the \textit{in vivo} regime~\cite{Cameron2008}."
and "Calculating effective sample sizes is discussed in detail in
Section~\ref{sec:Reg:subsec:comparison:subsubsec:samplesize}." (flattened
10/30 l. 4413; `sections/sec8_moodelComparison.tex:248`). The draft defines
$G$ as the number of *independent* clusters (subjects *in vivo*) to match that
paragraph, and defines $p$.
```latex
With few clusters, $\hat{V}$ is biased downwards and normal reference distributions
over-reject. A small-sample correction is therefore applied according to the number of
independent clusters $G$. For $G < 50$, the bias-reduced estimator (CR2) of Bell and
McCaffrey~\cite{BellMcCaffrey2002} is used, in which each $\hat{e}_n$ in
Eqn.~\ref{eqn:sandwich mean} is replaced by $(\mathbf{I} - \mathbf{H}_{nn})^{-1/2}\hat{e}_n$
with $\mathbf{H}_{nn} = \mathbf{X}_n(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}_n^\top$, and
$t$-statistics are referred to a $t$ distribution with Satterthwaite-type degrees of
freedom~\cite{Imbens2016}. For $G \geq 50$, the simpler CR1 scaling
$\tfrac{G}{G-1}\tfrac{N-1}{N-p}$, where $p$ is the number of columns of $\mathbf{X}$, with a
$t_{G-1}$ reference is used~\cite{Cameron2015}.
```
New bibliography entries: `BellMcCaffrey2002`, `Imbens2016`, `Cameron2015`.

### 2.8 [x] Mixed-effects Test 2 also reports $\tau^2_\alpha$ (~l. 1726, optional)
The Test 2 model has a random intercept as well as a random slope; the library
reports both $\tau^2_\alpha$ and $\tau^2_\beta$. Optionally add "and
$\tau^2_\alpha$" where $\tau^2_\beta$ is discussed.

### 2.9 [x] Permutation tests: one- and two-sided alternatives — `eqn:permutation_pval` (~l. 275)
The printed $p$-value is one-sided (upper tail). Add after the equation:
```latex
\noindent Eqn.~\ref{eqn:permutation_pval} gives the upper-tailed $p$-value, appropriate for
statistics that are large under the alternative, such as distances or squared statistics.
For a two-sided test of a signed statistic, the $p$-value is twice the smaller of the
upper- and lower-tailed values, truncated at one.
```
Library: `permutation_test(alternative="greater"|"less"|"two-sided")`.

### 2.10 [x] Bootstrap interval method (~l. 259, optional)
No change required: the library default is now the percentile interval
printed in the manuscript. Optionally add: "Bias-corrected and accelerated
(BCa) intervals~\cite{Efron1987} may be preferred for markedly skewed bootstrap
distributions." (`Efron1987` would be a new entry.)

### 2.11 [x] New subsection: assessing normality — insert after "Distributions in Deep Learning" (~l. 318), before `\subsection{Error Metrics}`
Requested 2026-10-10. The manuscript notes that normality is rarely checked
(~l. 233, 311–318) but does not say how to check it. The library implements
five tests (`analysis.normality`); the draft below covers what to test, how,
and how to interpret the result, and introduces each method.
```latex
\subsubsection{Assessing Normality}
\label{sec:perf:subsec:awareness:subsubsec:normality}
When an analysis relies on a normality assumption, the assumption should be checked on the
distribution that the analysis actually assumes to be normal, typically the residuals or the
paired differences $d_n = x_n - y_n$, separately for each metabolite and test-set stratum
(Section~\ref{...distributions...}). The check should combine a graphical assessment with a
formal test. A normal quantile--quantile (Q--Q) plot shows how the distribution departs from
normality, such as skewness, heavy tails or a small number of outliers, and descriptive
measures of skewness and excess kurtosis quantify the size of the departure. A formal test
then provides an objective decision rule for the null hypothesis that the sample was drawn
from a normal distribution.

Several tests are in common use, and they differ in the departures to which they are most
sensitive. The Shapiro--Wilk test~\cite{Shapiro1965} compares the ordered sample with the
expected order statistics of a normal distribution and has high power against a wide range of
alternatives~\cite{Razali2011}; it is recommended as the default. The Anderson--Darling
test~\cite{Anderson1952} is a weighted empirical distribution function test that gives
additional weight to the tails, which makes it well suited to detecting the heavy tails often
seen in residuals near the limit of detection; when the mean and variance are estimated from
the data, its critical values must be adjusted accordingly~\cite{Stephens1974}. The
D'Agostino--Pearson omnibus test~\cite{DAgostino1973,DAgostino1990} combines standardized
sample skewness and kurtosis into a single $\chi^2_2$ statistic and therefore also indicates
the nature of a departure. The Kolmogorov--Smirnov test~\cite{Kolmogorov1933,Smirnov1948,Massey1951}
is valid only against a fully specified distribution, that is, with a mean and variance fixed
independently of the data. When these parameters are estimated from the same sample, the
classical critical values are far too conservative, and the Lilliefors
correction~\cite{Lilliefors1967} must be used instead.

The interpretation of these tests depends strongly on sample size. With the evaluation-set
sizes recommended in this work ($n \geq 500$), the tests can detect departures from normality
that are too small to affect the validity of a parametric procedure, so a significant result
should be judged against the Q--Q plot and the magnitude of skewness and kurtosis rather than
treated as decisive on its own. Conversely, in small samples the tests have low power, and
failure to reject the null hypothesis does not establish that the data are normal
(Section~\ref{sec:perf:subsec:awareness}). When the same test is applied to many metabolites,
the number of tests should be accounted for, either by a multiplicity correction or by
reporting the results descriptively. The tests also assume independent observations, so
repeated spectra from the same subject should not be pooled as if they were independent
(Section~\ref{sec:Reg:subsec:comparison:subsubsec:samplesize}). Finally, using the outcome of
a normality test to choose between a parametric and a nonparametric analysis of the same data
alters the error rates of the subsequent analysis. Where possible, the analysis should
therefore be specified in advance on the basis of the expected error structure, with the
normality assessment reported as supporting evidence.
```
Fill in the `\ref{...distributions...}` placeholder with the label of
"Distributions in Deep Learning" (that subsubsection currently has no label;
add one).

New bibliography entries: `Lilliefors1967`, `Stephens1974`, `DAgostino1973`,
`DAgostino1990`, `Razali2011` (section 4). Already present: `Shapiro1965`,
`Anderson1952`, `Kolmogorov1933`, `Smirnov1948`, `Massey1951`.

Related library issue: `normality_test_suite` runs the Kolmogorov–Smirnov test
with the sample's own mean and SD, which is exactly the misuse the draft warns
against. It should either drop KS from the suite or require the parameters to
be supplied (decision pending).

---

## 3. Figures

### 3.1 [ ] Figure 5 (toy example): redesigned in the repository — update the caption
Regenerated 2026-10-10 (commit `8a2f955`) to follow the method-comparison
section for the *in silico* regime: Bland–Altman against the reference with
robust regression-based limits of agreement, confidence bands and
pre-specified acceptable limits; Passing–Bablok instead of Deming; and the
Silveira tests with the acceptability decision in panel D. Draft caption:
```latex
\caption{One paired \textit{in silico} dataset ($N = 150$, exact reference) answers four
different method-comparison questions.
\textbf{(A) Association:} the methods are strongly associated ($r = 0.988$,
$\rho_C = 0.939$), but the points lie above the identity line.
\textbf{(B) Bland--Altman disagreement} against the reference, with robust regression-based
bias and limits of agreement (median bias, normal-consistent MAD, Theil--Sen fits) and
95\,\% bootstrap confidence bands. The bias increases from about 0.4 to 1.6~mM across the
concentration range, and the limits of agreement lie outside the pre-specified acceptable
limits of $\pm 1$~mM.
\textbf{(C) Constant and proportional bias:} Passing--Bablok regression gives a slope of
1.118 [1.088, 1.151] and an intercept of 0.237 [0.021, 0.379]~mM (95\,\% CIs), indicating
both proportional and constant bias; the bootstrap-calibrated CUSUM test does not indicate
departure from linearity.
\textbf{(D) Fit for the intended use:} Silveira Tests~1--3 (bootstrap intervals; Test~3 at the
Bonferroni-adjusted 97.5\,\% level after translation correction) reject equal accuracy,
equal precision and agreement with the identity line, and the limits of agreement fall
outside the acceptable limits. A strong association (A) therefore does not imply that the
methods are interchangeable (D).}
```
Numbers come from `assets/publication/examples/toy_outputs/analysis/figure05_*.csv`;
update them if the toy data are regenerated.

---

## 4. New bibliography entries
All *(verify)*.
```bibtex
@article{Phipson2010,
  author  = {Phipson, Belinda and Smyth, Gordon K.},
  title   = {Permutation {P}-values Should Never Be Zero: Calculating Exact {P}-values When Permutations Are Randomly Drawn},
  journal = {Statistical Applications in Genetics and Molecular Biology},
  volume  = {9}, number = {1}, pages = {Article 39}, year = {2010},
  doi     = {10.2202/1544-6115.1585}
}
@book{Hajek1999,
  author    = {H{\'a}jek, Jaroslav and {\v{S}}id{\'a}k, Zbyn{\v{e}}k and Sen, Pranab K.},
  title     = {Theory of Rank Tests}, edition = {2nd},
  publisher = {Academic Press}, address = {San Diego}, year = {1999}
}
@article{Bland1999,
  author  = {Bland, J. Martin and Altman, Douglas G.},
  title   = {Measuring agreement in method comparison studies},
  journal = {Statistical Methods in Medical Research},
  volume  = {8}, number = {2}, pages = {135--160}, year = {1999},
  doi     = {10.1177/096228029900800204}
}
@article{Linnet1993,
  author  = {Linnet, Kristian},
  title   = {Evaluation of regression procedures for methods comparison studies},
  journal = {Clinical Chemistry},
  volume  = {39}, number = {3}, pages = {424--432}, year = {1993}
}
@article{York2004,
  author  = {York, Derek and Evensen, Norman M. and Mart{\'i}nez, Margarita L{\'o}pez and De Basabe Delgado, Jon{\'a}s},
  title   = {Unified equations for the slope, intercept, and standard errors of the best straight line},
  journal = {American Journal of Physics},
  volume  = {72}, number = {3}, pages = {367--375}, year = {2004},
  doi     = {10.1119/1.1632486}
}
@article{Viechtbauer2005,
  author  = {Viechtbauer, Wolfgang},
  title   = {Bias and Efficiency of Meta-Analytic Variance Estimators in the Random-Effects Model},
  journal = {Journal of Educational and Behavioral Statistics},
  volume  = {30}, number = {3}, pages = {261--293}, year = {2005},
  doi     = {10.3102/10769986030003261}
}
@article{BellMcCaffrey2002,
  author  = {Bell, Robert M. and McCaffrey, Daniel F.},
  title   = {Bias reduction in standard errors for linear regression with multi-stage samples},
  journal = {Survey Methodology},
  volume  = {28}, number = {2}, pages = {169--181}, year = {2002}
}
@article{Imbens2016,
  author  = {Imbens, Guido W. and Koles{\'a}r, Michal},
  title   = {Robust Standard Errors in Small Samples: Some Practical Advice},
  journal = {The Review of Economics and Statistics},
  volume  = {98}, number = {4}, pages = {701--712}, year = {2016},
  doi     = {10.1162/REST_a_00552}
}
@article{Cameron2015,
  author  = {Cameron, A. Colin and Miller, Douglas L.},
  title   = {A Practitioner's Guide to Cluster-Robust Inference},
  journal = {Journal of Human Resources},
  volume  = {50}, number = {2}, pages = {317--372}, year = {2015},
  doi     = {10.3368/jhr.50.2.317}
}
@article{Rousseeuw1993,
  author  = {Rousseeuw, Peter J. and Croux, Christophe},
  title   = {Alternatives to the Median Absolute Deviation},
  journal = {Journal of the American Statistical Association},
  volume  = {88}, number = {424}, pages = {1273--1283}, year = {1993},
  doi     = {10.1080/01621459.1993.10476408}
}
@article{Sen1968,
  author  = {Sen, Pranab Kumar},
  title   = {Estimates of the Regression Coefficient Based on {K}endall's Tau},
  journal = {Journal of the American Statistical Association},
  volume  = {63}, number = {324}, pages = {1379--1389}, year = {1968},
  doi     = {10.1080/01621459.1968.10480934}
}
@article{Theil1950,
  author  = {Theil, Henri},
  title   = {A rank-invariant method of linear and polynomial regression analysis, {I}, {II}, {III}},
  journal = {Proceedings of the Royal Netherlands Academy of Sciences, Series A},
  volume  = {53}, pages = {386--392, 521--525, 1397--1412}, year = {1950}
}
@article{Lilliefors1967,
  author  = {Lilliefors, Hubert W.},
  title   = {On the {K}olmogorov-{S}mirnov Test for Normality with Mean and Variance Unknown},
  journal = {Journal of the American Statistical Association},
  volume  = {62}, number = {318}, pages = {399--402}, year = {1967},
  doi     = {10.1080/01621459.1967.10482916}
}
@article{Stephens1974,
  author  = {Stephens, Michael A.},
  title   = {{EDF} Statistics for Goodness of Fit and Some Comparisons},
  journal = {Journal of the American Statistical Association},
  volume  = {69}, number = {347}, pages = {730--737}, year = {1974},
  doi     = {10.1080/01621459.1974.10480196}
}
@article{DAgostino1973,
  author  = {D'Agostino, Ralph and Pearson, E. S.},
  title   = {Tests for Departure from Normality. {E}mpirical Results for the Distributions of $b_2$ and $\sqrt{b_1}$},
  journal = {Biometrika},
  volume  = {60}, number = {3}, pages = {613--622}, year = {1973},
  doi     = {10.1093/biomet/60.3.613}
}
@article{DAgostino1990,
  author  = {D'Agostino, Ralph B. and Belanger, Albert and D'Agostino, Ralph B., Jr.},
  title   = {A Suggestion for Using Powerful and Informative Tests of Normality},
  journal = {The American Statistician},
  volume  = {44}, number = {4}, pages = {316--321}, year = {1990},
  doi     = {10.1080/00031305.1990.10475751}
}
@article{Razali2011,
  author  = {Razali, Nornadiah Mohd and Wah, Yap Bee},
  title   = {Power comparisons of {S}hapiro-{W}ilk, {K}olmogorov-{S}mirnov, {L}illiefors and {A}nderson-{D}arling tests},
  journal = {Journal of Statistical Modeling and Analytics},
  volume  = {2}, number = {1}, pages = {21--33}, year = {2011}
}
```
Possibly needed depending on decisions D1/D2: `Durbin1973`, `Stute1997`,
`Stute1998`, `Owen2007` (see below).

---

## 5. [ ] Supporting evidence for new numerical claims

Added 2026-11-02. Implementing the library produced numerical values the
manuscript did not have before (e.g. 1.4826 and ~81 % in 1.11, the 7.4 %
type-I error behind 2.1). Every number that ends up in the manuscript needs
support appropriate to where it came from. This section is a brief for
drafting that support; it is written so that a fresh session with access to
this repository can do the work.

### 5.1 Policy: three kinds of numbers

**A. Standard mathematical facts — cite or derive inline; no appendix.**
Values that follow from textbook results and that a reader can check
in one line. Give the closed form in the text (as the drafts already do,
e.g. $1/\Phi^{-1}(0.75)$) plus a citation. An appendix proof is warranted
only for a result that is new or not readily found in the literature.

**B. Simulation results — supplementary material, or keep the claim
qualitative.** Rates and ratios that came from Monte Carlo runs of the
library (type-I errors, rejection rates, design-effect discrepancies). A
reviewer cannot check them without the setup. If such a number is quoted
in the manuscript, it needs (i) an entry in a supplementary section and
(ii) a script in the repository that reproduces it exactly. If no
supplement is wanted, phrase the claim qualitatively ("anti-conservative",
"strongly conservative") and cite the literature instead.

**C. Toy-example outputs — code and data availability statement.**
Values computed by the example scripts on the toy data (Figure 5 caption).
They are reproducible from `assets/publication/examples/` and the CSVs in
`toy_outputs/analysis/`. Cite an archived, tagged release (e.g. Zenodo DOI).

### 5.2 Inventory

"In draft text?" = whether the number appears in the LaTeX drafted in this
file (and therefore in the manuscript once that item is applied), as
opposed to only in the explanatory notes here. Check the current
manuscript before acting: the author may have added numbers from the notes.

| # | Claim | Kind | Tracking item | In draft text? | Source in repo |
|---|---|---|---|---|---|
| a | MAD consistency constant $1/\Phi^{-1}(0.75) \approx 1.4826$; median absolute deviation of a normal $= 0.6745\,\sigma$ | A | 1.11, 2.2 | yes | `method_comparison.bland_altman(scale="robust_sd")` |
| b | Unscaled-MAD limits cover ~81 %: $2\Phi(1.96 \times 0.6745) - 1 \approx 0.814$ | A | 1.11 | yes | — (closed form) |
| c | $E\lvert Z\rvert = \sqrt{2/\pi}$, hence the $\sqrt{\pi/2}$ factor for absolute-residual scale models | A | 2.2 | yes | `bland_altman(conditional=True)` |
| d | ECWAS null correlations: Wilcoxon–AntiSavage $\approx -0.87 \to -\sqrt{3}/2$; Ansari-Bradley–AntiSavage $\approx -0.34$; Wilcoxon–Ansari-Bradley $= 0$ | A (exact for finite $n$ from the score functions) | 2.1 | yes | `src/mrsvalidate/analysis/distribution.py` (module docstring, ~l. 42) |
| e | ECWAS type-I error: $\chi^2_3$ critical value 7.4 %, permutation null 4.6 %, at nominal 5 % (100 + 100 obs.) | B | 2.1 | **no** (notes only) | `tests/unit/test_ecwas_critical_values.py::test_permutation_null_controls_type_one_error_where_chi2_does_not` (n = 100 per group, 4000 reps, seed 7, permutation critical value from 20 000 relabelings); CHANGELOG |
| f | Passing–Bablok CUSUM: classical Kolmogorov critical value 0–1 % rejection, wild bootstrap 2–5 %, at nominal 5 %; power against mild curvature at n = 200: 3 % vs 26 % | B | 2.4, D2 | **no** (D2 draft text says only "strongly conservative") | `tests/unit/test_passing_bablok_validation.py` (n = 60, 300 reps, uniform truth 1–10, error SD 0.3); `method_comparison.py` docstrings ~l. 1112, 1231; CHANGELOG |
| g | CUSUM bootstrap liberal (6–15 %) for unequal error variances at n ≥ 500 without $\lambda$; 4.6–5.8 % when $\lambda$ is supplied; 2026-11-04 rerun: 3–6 % with correct $\lambda$, 9–15 % *in silico* without $\lambda$, Theil–Sen 4.0–4.3 % (D2 table) | B | 2.4, D2 | partly: 2.4 draft says "can then be liberal" (qualitative) | `method_comparison.py` ~l. 1446 (warning text); CHANGELOG |
| h | Cluster-robust inference with 10 clusters: CR0 + normal reference rejects a true null 11–18 %; CR2 + Bell–McCaffrey 4–5 % | B | 2.7 | **no** (2.7 draft is qualitative) | `tests/unit/test_mixed_effects.py::test_cr2_bell_mccaffrey_size_closer_to_nominal_than_cr0_with_10_clusters` (docstring records a reference run: R = 2000, seed 123, 10 clusters × 6 obs.: CR0+z 0.110 / 0.175, CR1+t 0.061 / 0.120, CR2+BM 0.044 / 0.051 for intercept / slope); CHANGELOG |
| i | Old `eqn:deff_cc` gave deff 5.4 against a true ≈ 50 (≈ ten-fold overstatement of $N_\mathrm{eff}$) | B | 1.10 | **no** | `tests/unit/test_clustering.py::test_cross_classified_design_effect_equals_pooled_mean_variance_ratio_unbalanced` checks the corrected formula against the empirical variance ratio; the specific 5.4 vs 50 run is **not** stored |
| j | Panel-conditional inference rejects a true *population* null in 33 % of cases | B | D1 | **no** | `tests/unit/test_mixed_effects.py::test_mixed_test3_panel_mode_rejects_more_under_population_null` and neighbouring `..._panel_mode_size_...` / `..._population_mode_size_...` tests; the 33 % run itself is **not** stored |
| k | Translation correction leaves the Test 2 slope, SE and $p$ unchanged; intercept shifts by $-c(1-\hat\beta)$ | A (algebra; verified numerically) | 1.8 | algebra only | — (OLS invariance; can be shown in two lines) |
| l | Figure 5 caption values ($r = 0.988$, $\rho_C = 0.939$, PB slope 1.118 [1.088, 1.151], intercept 0.237 [0.021, 0.379] mM, BA bias 0.4 → 1.6 mM) | C | 3.1 | yes | `assets/publication/examples/compute_toy_analysis.py`; `toy_outputs/analysis/figure05_*.csv` |

Caveats for B items:
- The rates in the CHANGELOG and in this file came from development runs.
  Only (e) and (h) have their setup recorded in the repository (in the
  tests above). For (f), (g), (i) and (j) the repository has the test
  design but not the exact run that produced the quoted numbers.
  Regenerating them will give slightly different values; **use the
  regenerated values** in the manuscript and update the notes here.
- Unit tests use small replicate counts to stay fast. Publication runs
  need enough replicates that the Monte Carlo standard error is small
  relative to the differences claimed: SE of a rate $p$ over $R$ replicates
  is $\sqrt{p(1-p)/R}$ (≈ 0.22 percentage points for $p = 0.05$,
  $R = 10\,000$). Report it.

### 5.3 What to produce

1. **Decide per B item** whether the number is quoted (→ steps 2–3) or the
   claim is kept qualitative (→ no further work beyond a citation). Ask the
   author; default recommendation: quote (e) and (h), which directly
   justify library defaults the manuscript describes (2.1, 2.7); keep
   (i) and (j) qualitative unless the author wants them.
2. **Reproduction scripts** in a new directory
   `assets/publication/simulations/`, one script per quoted claim, following
   the style of `assets/publication/examples/compute_toy_analysis.py`:
   fixed seed, replicate count as a constant at the top, results written to
   CSV (`simulations/outputs/<claim>.csv`) with the rate, its Monte Carlo SE,
   and the full parameter set. Call the public library API, not private
   helpers, except where the test already does (e.g. CR0 in (h)). Reuse the
   data-generating models from the tests named in the inventory.
3. **Supplementary section** (LaTeX), e.g.
   `\section{Simulation studies supporting the methodological choices}`,
   one `\subsection` per quoted claim, each with: purpose (which manuscript
   statement it supports, by `\ref`); data-generating model with all
   parameters (sample sizes, error distributions, variance ratios, cluster
   structure); estimator(s) and critical values compared; number of
   replicates and seed; result table with Monte Carlo SEs; the script path.
   Keep each to a short paragraph plus a table. The manuscript text then
   cites it ("… 7.4 % at nominal 5 %; Supplementary Section~S\ref{…}").
4. **Inline derivations / citations for A items** (a)–(d), (k): check the
   drafted text in 1.8, 1.11, 2.1, 2.2 gives the closed form; add citations:
   Rousseeuw & Croux 1993 for the MAD constant (now cited in the 1.11 text;
   entry in section 4); Bland & Altman 1999 for the $\sqrt{\pi/2}$ absolute-residual
   scaling (already in section 4); Hájek, Šidák & Sen 1999 for the
   rank-score correlations (already in section 4).
5. **Code and data availability statement** for C items: name the
   repository, the tagged release used for the figures, and the archive
   DOI (placeholder until the release is cut). Check the journal's
   required heading.
6. Tick this section and the affected items here when done.

---

## Open decisions

**D1. [x] What the model-level uncertainty should include (mixed-effects Tests 1–3).** *Decided 2026-10-10: panel-conditional (default), with an optional population mode for Test 3; fixed effects are metabolite-balanced (equal-weight) means.*
The cluster bootstrap and sandwich estimators resample spectra/subjects only,
so inference is *conditional on the metabolite panel*: the between-metabolite
spread ($\tau^2$) does not enter the uncertainty of $\alpha$ and $\beta$.
That is correct if the claim is about this panel, and anti-conservative if it
is about a population of metabolites (33 % rejection of a true population null
in simulation). The manuscript's wording ("averaged over the metabolite
distribution", $u_i \sim \mathcal{N}(0,\tau^2)$) currently implies the latter.
Clarified 2026-10-10: these analyses compare *fitting methods* over the joint
set of metabolites they estimate, which is the panel-conditional reading.
Draft for the panel-conditional reading (recommended):
```latex
The fixed effects $\alpha$ and $\beta$ summarize agreement over the evaluated metabolite
panel, which is fixed by the basis set rather than sampled from a population of metabolites.
Inference is therefore conditional on that panel: the bootstrap and sandwich estimators
account for sampling variability across spectra (or subjects), while $\tau^2_\alpha$ and
$\tau^2_\beta$ are reported as descriptive measures of inter-metabolite heterogeneity.
```
Draft for the population reading (if chosen): replace the last sentence of
2.6 with resampling of both metabolites and spectra (two-way "pigeonhole"
bootstrap~\cite{Owen2007}) and note that the effective number of units is then
the number of metabolites.

**D2. [ ] Calibration of the Passing–Bablok CUSUM test.** The asymptotic
Kolmogorov critical value ignores that the line is estimated from the same
data, which makes the test very conservative (0–1 % rejection at nominal 5 %)
and weak: at $n = 200$ it detects mild curvature in 3 % of datasets, against
26 % for the bootstrap. The library now offers a wild-bootstrap calibration
that refits the line in every replicate (`cusum_method="bootstrap"`). In
validation it gives 2–5 % at nominal 5 %, near the best level attainable for
this discrete statistic, across homoscedastic, heteroscedastic and skewed
errors for $n$ = 50–1000. Exception: when the two methods have unequal error
variances and $n \geq 500$, flipping residuals perpendicular to the line is
liberal (6–15 %). Supplying the error-variance ratio
(`cusum_bootstrap_variance_ratio`, the same $\lambda$ as for Deming) restores
calibration (4.6–5.8 %). *Decided 2026-10-10: the bootstrap is the library
default; without a supplied $\lambda$ and with $n \geq 500$ it attaches a
warning recommending one. Use the bootstrap sentence below in section 2.4.*
Calibration sentence for the asymptotic version (for reference only):
```latex
Linearity is rejected when $H$ exceeds the $(1-\alpha)$ quantile of the Kolmogorov
distribution (1.36 at $\alpha = 0.05$)~\cite{Passing1983}; because the line is estimated
from the same data, this criterion is conservative.
```
Calibration sentence for the bootstrap-calibrated version:
```latex
Because the regression line is estimated from the same data, the null distribution of $H$
is not the Kolmogorov distribution~\cite{Durbin1973}, and the classical critical value is
strongly conservative. Critical values are therefore obtained by a wild
bootstrap under the fitted linear model~\cite{Stute1998}: each residual is multiplied by a
random sign along the direction conjugate to the fitted line under the error-variance ratio
$\lambda$ (vertical for an exact reference, $\lambda = 0$),
added back to its fitted position, the Passing--Bablok line is re-estimated for each
resample, and $H$ is recomputed. The $p$-value follows Eqn.~\ref{eqn:permutation_pval} with
the bootstrap replicates in place of permutations.
```

*Update 2026-11-04: how $\lambda$ is obtained, and recommendations.* The library
does not estimate $\lambda$; it is user-supplied (`cusum_bootstrap_variance_ratio`,
$\lambda = \sigma^2_{\text{ref}}/\sigma^2_{\text{DL}}$, same convention as Deming), and `None`
is equivalent to $\lambda = 1$ (perpendicular flips). It is not identifiable
from single paired measurements. The 0–1 % figure is the classical
Kolmogorov critical value; the 6–15 % figure is the bootstrap with the wrong
$\lambda$, which grows with $n$. Simulation (truth uniform 1–10, unit slope,
DL error SD 0.5 or 0.25 mM, nominal 5 %, 199 bootstrap replicates, 300–400
reps per cell, MC SE about 1.1–1.3 %):

| n | True setting | $\lambda$ supplied | Rejection |
|---|---|---|---|
| 500 / 1000 | *in silico* ($\lambda = 0$) | none (PB) | 9.0 % / 15.3 % |
| 1000 | *in silico* | $\approx 0$ (PB, 0.01) | 5.7 % |
| 500 / 1000 | *in silico* | Theil–Sen ($\lambda = 0$ fixed) | 4.0 % / 4.3 % |
| 500 / 1000 | $\lambda = 0.25$ | none | 5.2 % / 8.3 % |
| 500 / 1000 | $\lambda = 0.25$ | correct | 3.0 % / 3.0 % |
| 1000 | $\lambda = 0.25$ | $\lambda/4$ to $2\lambda$ | 3.7–5.0 % |
| 500 / 1000 | $\lambda = 4$ | none | 6.0 % / 12.3 % |
| 500 / 1000 | $\lambda = 4$ | correct | 4.8 % / 6.2 % |
| 1000 | $\lambda = 4$ | $\lambda/2$ / $2\lambda$ / $4\lambda$ | 6.3 % / 2.8 % / 3.3 % |
| 1000 | $\lambda = 1$ | correct | 5.8 % |

Correct $\lambda$: 3–6 % throughout. Overestimating errs conservative,
underestimating by 4× errs liberal. Recommendations (use the 2.4 text above):
1. Bootstrap only; never the classical critical value.
2. *In silico*: Theil–Sen ($\lambda = 0$ built in). Do not run the PB CUSUM
   without $\lambda = 0$.
3. *In vivo*: always supply the Deming $\lambda$ (reference from absolute
   CRLB, DL from test–retest / MC / calibrated uncertainty, both in mmol²/L²).
4. Uncertain $\lambda$: run at $\lambda/2$, $\lambda$, $2\lambda$ and report
   the largest $p$-value.

Library follow-up: see 2.4 ("Library (to do)").

**D3. [ ] Figure 5 redesign.** *Decided and implemented 2026-10-10; see section 3.1.*
