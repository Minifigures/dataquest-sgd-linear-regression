# Stochastic Gradient Descent on Linear Regression

Predicting university gym occupancy from 62,185 observations taken every ten minutes over a year. The subject is the optimiser rather than the problem.

Dataquest guided project, Data Scientist in Python path.

## Scaling is the whole lesson

Same optimiser, same `eta0 = 1e-4`, same 50 epochs. The only difference is whether features were standardised first:

| | RMSE | MAE | R-squared | Largest coefficient |
|---|---|---|---|---|
| Baseline: predict the training mean | 22.734 | 18.376 | -0.0000 | |
| **Closed-form OLS (reference)** | **15.830** | **12.090** | **0.5152** | 1.317e+01 |
| **SGD, unscaled** | **diverged** | **diverged** | **diverged** | **5.171e+12** |
| SGD, scaled | 15.844 | 12.109 | 0.5143 | 9.2128 |

The unscaled run did not perform badly, it **diverged**: its largest prediction reached 4.467e+17 people in a gym whose observed maximum is 145, with coefficients at 5.17e+12. The scaled run, every other setting identical, recovers **99.8% of the closed-form R-squared**.

The cause is measurable: `timestamp` has standard deviation 24,211 and `is_holiday` has 0.051, a ratio of **477,917 to 1**. The gradient with respect to a feature scales with that feature's magnitude, so a step size stable along `timestamp` is five orders of magnitude too small for the binary flags, and one large enough to move the flags overflows along `timestamp`. Standardising makes the loss surface round so one step size works in every direction.

## The learning rate window is narrow and the library default is outside it

| eta0 | Test R-squared | Largest coefficient |
|---|---|---|
| 1e-7 | -0.6823 | 2.412 |
| 1e-6 | 0.5022 | 5.660 |
| 1e-5 | 0.5142 | 5.985 |
| **1e-4** | **0.5143** | 9.213 |
| 1e-3 | 0.5112 | 20.129 |
| 1e-2 | **diverged** | overflow |
| 0.1 | **diverged** | overflow |
| 1.0 | **diverged** | overflow |

The usable range is about 1e-6 to 1e-3, and **scikit-learn's default `eta0=0.01` diverges on this data**. At 1e-7 the model is stable but has not moved far enough from zero in 50 epochs to beat the mean baseline (-0.68). Both failure modes are silent: nothing raises, and coefficients of 5e+12 look like a fitted model unless someone checks.

| Schedule (eta0 = 1e-4) | Test R-squared |
|---|---|
| constant | 0.5143 |
| **optimal** | **diverged** |
| invscaling | 0.5141 |
| adaptive | 0.5143 |

`optimal` is the trap worth knowing: it **ignores `eta0` entirely**, setting the rate to 1 / (alpha * (t + t0)), so with the default `alpha=1e-4` the opening step is enormous whatever was passed. Supplying `eta0` alongside `learning_rate='optimal'` looks like configuration and does nothing.

## Convergence, watched rather than assumed

| Epoch | Test MSE |
|---|---|
| 1 | 251.2267 |
| 10 | 251.0842 |
| 20 | 251.1319 |
| 60 | 250.9550 |
| **Closed-form OLS** | **250.5929** |
| Mean baseline | 516.8491 |

Most of the work happens in the first epoch. Epochs 20 to 60 buy 0.177 MSE, and a gap of 0.362 to the exact solution remains after 60 passes. SGD approaches the closed form asymptotically without reaching it, which is expected rather than a bug.

## At this scale SGD is a demonstration, not a tool

| | Fit time |
|---|---|
| Closed-form OLS | **6.29 ms** |
| SGD, 50 epochs | 211.17 ms |

**33.6 times slower** on 49,747 rows and 9 features, for a slightly worse answer. SGD earns its place when the matrix cannot be formed or inverted, or when data arrives as a stream. Neither applies here, and saying so is more useful than implying otherwise.

## Cyclical encoding helps; one-hot helps far more

| Feature set | Features | CV R-squared | std |
|---|---|---|---|
| A: raw, both `timestamp` and `hour` | 9 | 0.5092 | 0.0034 |
| B: cyclical sine/cosine pairs | 11 | 0.5212 | 0.0023 |
| **C: hour as 24 indicators** | 33 | **0.6068** | 0.0026 |

The textbook cyclical fix is worth **+0.012**; replacing it with 24 indicators is worth a further **+0.086**, seven times more. The daily curve is not a sine wave: it climbs through the morning, plateaus across afternoon and evening, then drops sharply overnight. One sine/cosine pair fits a single smooth oscillation; 24 indicators fit any shape. Cyclical encoding is the right default when the shape is roughly sinusoidal or there are too many levels to one-hot, and with 24 levels and 62,000 rows the parameters are clearly worth spending.

On the best feature set Ridge and SGD are indistinguishable (R-squared 0.6090 both), while a random forest reaches **0.8782**. That 0.27 gap is the non-linear share of the signal, unreachable by any linear model however well optimised.

## The number that matters most: the random split was overstating everything by 1.17 R-squared

Every figure above uses a random 80/20 split. This is a time series sampled every ten minutes, so consecutive rows are near-duplicates and a random split puts 17:00:11 in training and 17:10:14 in test. The model is graded on rows it has effectively already seen.

| Split | Ridge RMSE | Ridge R-squared | Baseline R-squared | Train mean | Test mean |
|---|---|---|---|---|---|
| Random | 14.215 | **0.6090** | -0.0000 | 29.08 | 29.04 |
| **Chronological** | 18.320 | **-0.5575** | -1.0293 | 32.05 | 17.16 |

**Optimism: 1.1665 R-squared.** Trained on 2015-08-14 to 2016-10-26 and tested on the following five months, the model scores worse than predicting a constant. It is not a weak forecaster, it is an actively harmful one.

The last two columns explain it: the training period averaged 32.05 people, the test period 17.16. The gym got substantially quieter and the model learned a level that no longer applies. The mean baseline also collapses to -1.0293 on that split, confirming genuine distribution shift rather than a modelling failure.

Anyone citing 0.6090 as this model's forecasting accuracy would be out by more than a full unit of R-squared. The random-split figure validly measures one thing only, interpolating occupancy inside a period the model has already seen, and nobody wants that.

## Method

1. Load, profile, and catch that `timestamp` is a deterministic function of `hour` (correlation 1.0, seconds since midnight).
2. Quantify the scale disparity across raw predictors.
3. EDA, especially occupancy by hour, which decides the encoding.
4. Build three feature sets: raw, cyclical sine/cosine, and one-hot hour.
5. **Run SGD unscaled and scaled with identical settings**, against a closed-form reference.
6. Sweep `eta0` to the divergence boundary and compare all four learning-rate schedules.
7. Track loss per epoch with `partial_fit` instead of trusting `max_iter`.
8. Time the closed form against SGD and state when each is appropriate.
9. Compare feature sets on identical folds.
10. **Re-evaluate on a chronological split** and report the optimism.

## Limitations

- The headline 0.6090 is interpolation, not forecasting. The chronological -0.5575 is the deployment number and the one to take away.
- The distribution shift is detected, not diagnosed. Whether occupancy fell for a semester, facility or policy reason cannot be determined from these columns, and the fix (rolling retraining, or a level/trend term) was not implemented.
- `eta0` was swept on the test set. That section demonstrates the stability window; it is not a legitimate model-selection procedure, which belongs in cross-validation on the training fold.
- The random forest was an untuned reference for the non-linear ceiling, so 0.8782 is a floor on what a non-linear model could reach.
- Observations are ten minutes apart and strongly autocorrelated, so the effective sample size is far below 62,185 and the cross-validation standard deviations are optimistic.
- `timestamp` and `hour` are redundant by construction; feature set A keeps both deliberately to make the scaling problem stark. That is a pedagogical choice, not a recommendation.
- 14.11% of observations are zero occupancy, mostly overnight, so the target is mildly zero-inflated. A censored or two-stage specification would respect the boundary that plain least squares happily predicts through.

## Stack

Python, pandas, NumPy, scikit-learn (SGDRegressor, LinearRegression, Ridge, StandardScaler, Pipeline, RandomForestRegressor, KFold, partial_fit), matplotlib.

## Files

- `notebook.ipynb`: the full analysis with executed outputs and figures.