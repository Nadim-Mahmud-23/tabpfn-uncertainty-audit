# Cover Letter

Nadim Mahmud Dipu
Brac University, Dhaka, Bangladesh
nadim.mahmud.dipu@g.bracu.ac.bd
ORCID: 0009-0005-8767-1636

Date: _______________

To the Editor-in-Chief
International Journal of Approximate Reasoning
Elsevier

Dear Editor-in-Chief,

I am pleased to submit my manuscript, **"When does group-conditional conformal
calibration help? Subgroup coverage, small-group failure, and an uncertainty audit
of TabPFN,"** for consideration as a research article in the *International Journal of
Approximate Reasoning*.

Split conformal prediction equips any classifier with finite-sample, distribution-free
*marginal* coverage, but marginal validity is silent about how miscoverage is
distributed across protected subgroups — precisely the property that matters when
prediction sets gate consequential decisions. This paper studies, both empirically and
through an exact finite-sample analysis, **when group-conditional (Mondrian) calibration
repairs subgroup miscoverage and when it makes it worse.** The central contributions are:

1. **An exact, pre-deployment null model** for the realized per-group coverage gap. Using
   the Beta law of split-conformal coverage together with a binomial test-measurement
   layer, the full distribution of the gap statistic *under zero structural bias* is
   computed from the target level, the per-group calibration counts, and the test stratum
   sizes alone — before any group-conditional calibration is run. It decomposes any
   observed gap into structural bias and sampling noise and yields an *a priori* test of
   whether stratification can help on a given axis.

2. **A controlled demonstration that plain Mondrian calibration can *increase* the realized
   disparity** it is meant to remove, on a real protected attribute (race) whose smallest
   calibration stratum holds only ~84 points, and a quantitative account of the
   marginal-versus-Mondrian crossover as a function of calibration support — validated by a
   calibration-size sweep and by simulation.

3. **A matched-protocol uncertainty audit of the tabular foundation model TabPFN**
   (calibration, conformal efficiency, and group-conditional coverage) against
   gradient-boosted-tree and MLP baselines, including a temperature-scaled-GBDT comparison
   that isolates the recalibration component of the deployed recipe.

The work is squarely within the scope of *IJAR*: it concerns the quantification, validity,
and finite-sample behaviour of predictive uncertainty, and it contributes to the conformal
prediction literature that the journal regularly hosts. All claims are supported by an
open, fully scripted pipeline with committed per-seed result files and a verification
script that recomputes every reported number.

I confirm that this manuscript is original, has not been published previously, and is not
under consideration for publication elsewhere. There is a single author. The author
declares no competing interests, and no funding was received for this research. All code,
data-loading scripts, and per-seed result files are publicly available, and an archival
copy will be deposited with a persistent identifier prior to publication.

A list of suggested reviewers is provided separately. Thank you for considering this
submission; I look forward to the reviewers' feedback.

Sincerely,

Nadim Mahmud Dipu
Brac University, Dhaka, Bangladesh
