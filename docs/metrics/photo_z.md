# Photo-z PDF and Local-Calibration Metrics

These diagnostics follow the companion photo-z papers and are designed to
separate whole-sample PDF quality from calibration among photometrically
similar galaxies.

## PIT and PIT Cramér–von Mises

For each test object with reference redshift $z_s$ and predictive CDF $F_i$,

$$
u_i = F_i(z_{s,i}),\qquad
\mathrm{PITCvM} = \int_0^1 [\hat F_{\rm PIT}(u)-u]^2\,du .
$$

For $N$ independent uniform PIT values the expected value is $1/(6N)$;
`pit_cvm(..., normalized=True)` divides by this reference.

```python
from torchregress.metrics import pit_cvm

stat = pit_cvm(pit, normalized=True)
```

## Local-W1

For each object, select its $k=100$ nearest neighbours in standardized
photometric feature space (neighbours in photometry, not sky) and compare the
neighbour PIT distribution with Uniform(0,1) using the 1-Wasserstein distance;
report the median over objects.  Smaller values mean better local calibration.
At $k=100$ the finite-sample reference is approximately 0.028, not a strict
lower bound.

```python
from torchregress.metrics import pit_local_w1

local = pit_local_w1(photometry, pit, k=100)
```

## Binned CRPS

For a PDF stored as probability masses on fixed bin centers, convert the CDF to
a piecewise-linear function between edges and integrate exactly:

$$
\mathrm{CRPS} = \left\langle \int [F_i(z)-\mathbf{1}(z\ge z_{s,i})]^2\,dz \right\rangle .
$$

`crps_from_binned_cdf` accepts CDF values at the bin edges (use
`binned_cdf(probabilities)` to construct them) and evaluates each object's
integral in closed form, including below/above the support.

```python
from torchregress.metrics import binned_cdf, crps_from_binned_cdf

cdf = binned_cdf(probabilities)           # [N, K+1]
score = crps_from_binned_cdf(edges, cdf, z_true)
```

## PDF width and paired uncertainty

`pdf_width_68(samples, redshift)` is the median central-68% width divided by
$1+z$ (the paper's $W_{68}$).  `paired_bootstrap_ci(a, b)` returns the paired
percentile interval for the mean difference; the default is the 16th–84th
percentile (68% interval).  This is test-sample uncertainty only; it does not
include retraining variation.  Use [PIT diagnostics](distribution.md) for
standard histogram/KS statistics and [calibration metrics](calibration.md) for
post-hoc calibration.

## References

| # | Reference |
|:-:|:----------|
| 1 | Chang et al. (2026), companion photo-z paper; PIT CvM, local-W1, W68 and binned CRPS conventions. |
| 2 | Polsterer, K. L. et al. "Photometric redshifts with CRPS." 2016 (binned CRPS/PDF scoring). |
| 3 | D'Isanto, A. & Polsterer, K. L. "Photometric redshift estimation via deep learning." A&A 2018 (PIT/PDF diagnostics). |
