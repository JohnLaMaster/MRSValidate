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

### 1.3 [ ] Coefficient of determination: missing square — `eqn:r2` (~l. 434)
```latex
\\ \textnormal{ and } SS_{tot} &= \sum_i (y_i - \Bar{y})^2
```

### 1.4 [?] Power SNR — `eqn:power snr` (~l. 69)
Agreed so far: the second line must square the magnitude (it currently sums
$|y|$, i.e. mean magnitude, not power):
```latex
&= \sqrt{\frac{1}{\sigma_{noise}^2}\cdot \frac{1}{Ns}\sum_{n=1}^{Ns}\left(\sqrt{\Re(y)^2 + \Im(y)^2}\right)^2}
```
Still open:
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

### 1.5 [ ] Deming slope written for the reverse regression — `eqn:deming_slope` (~l. 1631)
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

### 1.6 [ ] Contradictory definitions of the error-variance ratio $\lambda$
`eqn:deming_lambda` defines $\lambda=\sigma_y^2/\sigma_x^2$ (reference/DL), and
the limiting cases in the Deming text ($\lambda=1$ orthogonal, $\lambda\to0$
OLS of $x$ on $y$) are correct for that definition. Make the following agree with it:
- Deming text after `eqn:deming_objective` (~l. 1630): "$\lambda = \sigma_x^2/\sigma_y^2$" → $\sigma_y^2/\sigma_x^2$.
- Silveira table, Test 2 structural $H_0$: $\lambda=\mathrm{V}[\epsilon]/\mathrm{V}[\delta]$ → $\mathrm{V}[\delta]/\mathrm{V}[\epsilon]$ (the Test 2 text, ~l. 1588, already has $\mathrm{V}[\delta]/\mathrm{V}[\epsilon]$; $\delta$ is the error of $y$).
- Mixed Test 3 text after `eqn:me_test3` (~l. 1736): $\lambda_{i,n}=\sigma^2_{\mathrm{DL}}/\sigma^2_{\mathrm{ref}}$ → $\sigma^2_{\mathrm{ref},i,n}/\sigma^2_{\mathrm{DL},i,n}$.

### 1.7 [ ] Silveira Test 1 text swaps $x$ and $y$ (~l. 1586)
"regressing the pairwise differences $y_i - x_i$ on the mean-centered
reference values $x_i - \bar{x}$" → "regressing $x_i - y_i$ on the
mean-centered reference values $y_i - \bar{y}$", matching the table and
`eqn:me_test1`. Correspondingly, "the regression intercept $\alpha$
corresponds directly to the mean difference $\bar{x} - \bar{y}$".

### 1.8 [ ] Translation correction cannot change the Test 2 result (~l. 1588, ~l. 1726)
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
reword the same way.

### 1.9 [ ] Typo in `eqn:me_test3` (~l. 1733)
`\epsilon{i,n}` → `\epsilon_{i,n}`.

### 1.10 [ ] Cross-classified design effect pairs each ICC with the wrong cluster size — `eqn:deff_cc`
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

### 1.11 [ ] Robust limits of agreement cover ~81 %, not 95 % — Bland–Altman, "Robust Estimation of Bias" (~l. 1517)
$\widetilde\sigma$ (`eqn:np variance`) is the unscaled median absolute
deviation, which is $0.674\,\sigma$ for normal data, so
$\widetilde d \pm 1.96\,\widetilde\sigma_d$ covers about 81 % of differences.
Add after the sentence introducing $\widetilde{\sigma}_d$:
```latex
For the limits of agreement, $\widetilde{\sigma}_d$ is multiplied by the
normal-consistency constant $1/\Phi^{-1}(0.75) \approx 1.4826$, so that
$\widetilde{d} \pm 1.96 \times 1.4826\,\widetilde{\sigma}_d$ covers 95\,\% of
differences under normality; without this constant the coverage is
approximately 81\,\%. Where no distributional form is assumed, the limits may
instead be taken as the empirical 2.5\,\% and 97.5\,\% quantiles of $d_n$.
```
Note: `eqn:np variance` defines $\widetilde\sigma=\sqrt{\mathrm{med}((\widetilde\mu-x_n)^2)}$,
which equals the MAD for odd $N$ but not exactly for even $N$; the library's
Bland–Altman uses the MAD, $\mathrm{med}|d_n-\widetilde d|$, directly. Consider
defining $\widetilde\sigma$ as the MAD.
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

### 1.13 [ ] Inconsistent sample-variance normalisation in the Monte Carlo section (~l. 864, 982, 986)
$\hat\sigma^2_{\mathrm{DL}}$ uses $1/(K-1)$ while $\widehat{RMSE}$ uses
$1/K$, so $\widehat{RMSE}^2 = \widehat{\mathrm{Bias}}^2 + \hat\sigma^2$ holds
only approximately (exactly with $1/K$ in both). Either use $1/K$ in the
variance used for the decomposition, or state that the identity holds up to
the factor $(K-1)/K$.

### 1.14 [ ] Opposite error sign convention in the multivariate section (~l. 918)
$\mathbf{e} = \mathbf{c} - \hat{\mathbf{c}}$ is reference minus prediction,
the reverse of the $x - y$ convention used elsewhere. Harmless for $D^2$, but
for consistency use $\mathbf{e} = \hat{\mathbf{c}} - \mathbf{c}$.

### 1.15 [ ] Broken cross-reference (~l. 1327, ~l. 1523)
`\ref{subsubsec:Lin's rho}` → `\ref{sec:Eval:subsec:RegMetrics:subsubsec:Lin's rho}`.

### 1.16 [ ] ECWAS: max-type moments are not analytic (~l. 612)
"derived analytically from the null distributions as functions of sample size"
is true for $WI$, $AB$, $AS$, but not for $\max(AS,S)$ in `eqn:funato`. Covered
by the new paragraph in 2.1.

---

## 2. New text needed (library behavior the manuscript does not yet describe)

### 2.1 [ ] ECWAS critical values — new paragraph after `eqn:funato` (Distributional Similarity, "Hypothesis Tests for DL-Based Studies")
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

### 2.2 [ ] Bland–Altman: how the conditional models and confidence bands are computed — "Regression-Based Limits of Agreement" and "Confidence Intervals and Clinical Acceptability" (~l. 1510–1521)
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

### 2.3 [ ] Passing–Bablok confidence intervals — "Passing-Bablok Regression" (~l. 1617)
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

### 2.4 [?] Passing–Bablok CUSUM linearity test — "Passing-Bablok Regression" (~l. 1615)
The manuscript cites the test but does not define it or its critical value.
Text depends on the calibration decision (Open decision D2). Common part:
```latex
For the linearity test, each residual $e_i = x_i - (\hat{\beta}_0 + \hat{\beta}_1 y_i)$ is
assigned the score $r_i = \sqrt{n_-/n_+}$ if $e_i > 0$, $r_i = -\sqrt{n_+/n_-}$ if $e_i < 0$
and $r_i = 0$ otherwise, where $n_+$ and $n_-$ are the numbers of positive and negative
residuals, so that $\sum_i r_i = 0$. The scores are ordered by the position of each
observation along the fitted line, and the test statistic is
$H = \max_k \left|\sum_{i \le k} r_{(i)}\right| / \sqrt{n_+ + n_- + 1}$.
```
followed by one of the two calibration sentences in D2.

### 2.5 [ ] Silveira Test 3: analytic intervals — "Equivalence Testing" (~l. 1590–1592)
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

### 2.6 [ ] Mixed-effects Test 3: estimation procedure — "Mixed-Effects Equivalence Testing", after `eqn:me_test3` (~l. 1736)
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

### 2.7 [ ] Small-sample correction for the cluster-robust standard errors — after `eqn:sandwich mean` (~l. 1716)
The manuscript notes that enough clusters are needed (citing Cameron2008) but
gives no correction. Library default: CR2 with Bell–McCaffrey degrees of
freedom below 50 clusters, CR1 with $t(G-1)$ above. In simulation with 10
clusters, the uncorrected estimator with a normal reference rejected a true
null in 11–18 % of cases at nominal 5 %, against 4–5 % with CR2.
```latex
With few clusters, $\hat{V}$ is biased downwards and normal reference distributions
over-reject. A small-sample correction is therefore applied according to the number of
clusters $G$. For $G < 50$, the bias-reduced estimator (CR2) of Bell and
McCaffrey~\cite{BellMcCaffrey2002} is used, in which each $\hat{e}_n$ in
Eqn.~\ref{eqn:sandwich mean} is replaced by $(\mathbf{I} - \mathbf{H}_{nn})^{-1/2}\hat{e}_n$
with $\mathbf{H}_{nn} = \mathbf{X}_n(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}_n^\top$, and
$t$-statistics are referred to a $t$ distribution with Satterthwaite-type degrees of
freedom~\cite{Imbens2016}. For $G \geq 50$, the simpler CR1 scaling
$\tfrac{G}{G-1}\tfrac{N-1}{N-p}$ with a $t_{G-1}$ reference is used~\cite{Cameron2015}.
```
New bibliography entries: `BellMcCaffrey2002`, `Imbens2016`, `Cameron2015`.

### 2.8 [ ] Mixed-effects Test 2 also reports $\tau^2_\alpha$ (~l. 1726, optional)
The Test 2 model has a random intercept as well as a random slope; the library
reports both $\tau^2_\alpha$ and $\tau^2_\beta$. Optionally add "and
$\tau^2_\alpha$" where $\tau^2_\beta$ is discussed.

### 2.9 [ ] Permutation tests: one- and two-sided alternatives — `eqn:permutation_pval` (~l. 275)
The printed $p$-value is one-sided (upper tail). Add after the equation:
```latex
\noindent Eqn.~\ref{eqn:permutation_pval} gives the upper-tailed $p$-value, appropriate for
statistics that are large under the alternative, such as distances or squared statistics.
For a two-sided test of a signed statistic, the $p$-value is twice the smaller of the
upper- and lower-tailed values, truncated at one.
```
Library: `permutation_test(alternative="greater"|"less"|"two-sided")`.

### 2.10 [ ] Bootstrap interval method (~l. 259, optional)
No change required: the library default is now the percentile interval
printed in the manuscript. Optionally add: "Bias-corrected and accelerated
(BCa) intervals~\cite{Efron1987} may be preferred for markedly skewed bootstrap
distributions." (`Efron1987` would be a new entry.)

### 2.11 [ ] New subsection: assessing normality — insert after "Distributions in Deep Learning" (~l. 318), before `\subsection{Error Metrics}`
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

## Open decisions

**D1. What the model-level uncertainty should include (mixed-effects Tests 1–3).** *Decided 2026-10-10: panel-conditional (default), with an optional population mode for Test 3; fixed effects are metabolite-balanced (equal-weight) means.*
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

**D2. Calibration of the Passing–Bablok CUSUM test.** The asymptotic
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
calibration (4.6–5.8 %). **Pending decision:** which method is the library
default (currently the 1983 test; Figure 5 uses the bootstrap with the known
$\lambda$). Calibration sentence for the current asymptotic version:
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
$\lambda$ (perpendicular to the line when $\lambda$ is unknown and assumed equal to one),
added back to its fitted position, the Passing--Bablok line is re-estimated for each
resample, and $H$ is recomputed. The $p$-value follows Eqn.~\ref{eqn:permutation_pval} with
the bootstrap replicates in place of permutations.
```

**D3. Figure 5 redesign.** *Decided and implemented 2026-10-10; see section 3.1.*
