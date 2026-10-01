# FNPH Yaba — Data Cleaning Log

This file records every data quality issue found in the raw dataset, what
was decided about it, and why. Nothing gets silently deleted or changed
without an entry here — that's a graded requirement of the brief, not
just good practice.

Add a new entry every time you find and handle an issue in **any** table,
not just `admissions`.

---

## Table: `admissions.csv`

### Issue 1 — Inflated `length_of_stay_days` values

- **Found:** 20 rows where the recorded `length_of_stay_days` disagreed
  with the value calculated from `admit_date`/`discharge_date` by more
  than 5 days.
- **Pattern noticed:** almost every flagged row's recorded value is
  roughly **9× larger** than the date-calculated value — this looks like
  a systematic error (e.g. a unit or multiplier bug at data entry), not
  scattered random mistakes.
- **Threshold check:** re-ran the flag at thresholds of 1, 2, 3, 5, 10,
  20 and 50 days. The count stayed stable at 20 across the 5–50 range,
  confirming the 20 true errors sit far outside the range of normal
  date/time-of-day rounding noise (which only affects the low end of the
  range, e.g. 0–2 days).
- **Decision:** overwrote `length_of_stay_days` with the date-calculated
  value for those 20 rows only. All other rows left untouched.
- **Why:** the admit/discharge dates are ground truth; the recorded
  field appears to carry a systematic error, so the calculated value is
  more trustworthy than the recorded one.

### Issue 2 — Missing values in 3 columns

| Column                       | Missing count | % of rows |
| ---------------------------- | ------------- | --------- |
| `family_support_score`       | 103           | 1.20%     |
| `medication_adherence_score` | 181           | 2.11%     |
| `total_cost_ngn`             | 42            | 0.49%     |

- **Decision:** imputed using `sklearn.IterativeImputer`
  (MICE-style — Multiple Imputation by Chained Equations), using these
  columns as predictors: `family_support_score`,
  `medication_adherence_score`, `total_cost_ngn`, `age_at_admission`,
  `prior_admissions`, `severity_on_admission`, `severity_on_discharge`,
  `length_of_stay_days`, `seclusion_episodes`.
- **Settings:** `random_state=42` (reproducible — matches the project's
  `RANDOM_SEED`), `max_iter=10`.
- **Rounding:** all three imputed columns rounded to the nearest whole
  number using standard rounding (`.round(0)`, no directional bias),
  matching the whole-number format already used in the real recorded
  values for these columns.
- **Answer-key check:** `patients.segment_label` and
  `unit_quarterly_panel.true_efficiency_hidden` were **not** used as
  predictors, per the brief's rule that these may only be used for
  post-hoc checks (ARI/NMI, efficiency validation).
- **Verification:** re-ran `df.isna().sum()` after imputation — all
  three columns returned 0 missing values.
- **Output file:** saved as `admissions_cleaned.csv` — the original
  `admissions.csv` was left untouched so the raw source remains
  available for comparison.

---

---

## Table: `care_pathway_transitions.csv` — Absorbing chain modelling (Week 2)

### Issue — transpose bug in API output

- **Found:** `pandas.DataFrame.to_dict()` defaults to column-first
  orientation (`{column: {row: value}}`), not row-first. Since the
  transition matrix uses the same 10 state labels on both axes, this
  silently returned the transpose of the intended matrix, with no
  visible error — confirmed by comparing
  `transition_matrix["Acute inpatient"]["Community relapse"]` (wrongly
  non-zero) against `transition_matrix["Community relapse"]["Acute
inpatient"]` (correctly 0.3163) in the raw API output.
- **Fix:** added `orient="index"` to every `.to_dict()` call on the
  transition matrix, fundamental matrix, and absorption-probability
  matrix, forcing `{state_from: {state_to: value}}` ordering.
- **Verification:** re-checked the same two cells after the fix — the
  values were correctly swapped, confirming the matrix now reads in
  the intended direction.

### Finding — Lost to follow-up dominates every starting state

- Absorption probability into "Lost to follow-up" ranges from **35.4%
  to 43.1%** depending on starting state — the largest or
  second-largest outcome from every transient state, ahead of
  "Recovered and discharged from service" when starting from
  Community relapse specifically (43.1% vs 42.3%).
- Directly quantifies the board's stated concern about repeat spend
  without resolution; flagged for inclusion in the technical report
  and as a driver for the bed & staffing model.

### Sanity checks performed

- Every row of the transition matrix and the absorption-probability
  matrix sums to exactly 1.0.
- Expected months to absorption (7.71–8.90 months across starting
  states) are all positive and clinically plausible.
- Cross-validated the full pipeline (matrix, N, B, expected months)
  computed independently in a Colab notebook against the live API
  output — identical results, confirming the database-backed endpoint
  matches the standalone calculation.
- 1.41% of patients are censored (still mid-treatment, no absorption
  event yet as of the data extract) — expected, not an error, and
  excluded from skewing the chain since only observed moves are
  counted.

## Table: (add next table here as you clean it)

- **Issue:**
- **Found:**
- **Decision:**
- **Why:**

---

## Reminders while filling this in

- Never delete a row/value without recording why here first.
- Two columns are answer keys and must never be used as model inputs:
  `patients.segment_label` (only after clustering, for ARI/NMI) and
  `unit_quarterly_panel.true_efficiency_hidden` (only for the final
  efficiency check).
- If a value looks wrong but you're not sure why, write down your best
  guess at the cause — a documented guess is worth more than silence.

  ***

## Forecasting: monthly admissions (Week 3)

- **Series:** 36 months (Jan 2023–Dec 2025), no gaps. Held out final 6
  months as test set.
- **Seasonality confirmed** via decomposition: real yearly pattern
  present, though built on only 3 observed cycles (a stated limitation
  for a 12-month seasonal period).
- **Two anomalous months found:** October 2025 (actual 191 vs. ~250
  expected by every method — a genuine one-off drop not explained by
  the seasonal pattern) and December 2025 (actual 256 vs. ~208–210
  expected — the seasonal "December dip" was built from only 2 prior
  Decembers and didn't hold a third time).
- **Methods compared:** naive seasonal baseline, exponential smoothing,
  ARIMA (auto-selected order, D=1 set manually due to short-series
  limitation in the automatic seasonal test), XGBoost (lag-1, lag-12,
  and calendar-month features), LSTM (12-month lookback, single-step
  recursive forecast).
- **Result:** ARIMA and XGBoost tied as the strongest methods (MAE
  26.24 vs 27.65, RMSE 33.58 vs 33.58); LSTM and exponential smoothing
  underperformed both; all four genuinely beat the naive baseline
  (MAE 39.00), confirming the modelling effort added real value.
- **Production choice: XGBoost**, chosen over the near-identical ARIMA
  because it extends naturally to per-ward forecasting (via a unit_id
  feature) without needing a separate model per ward.
- **Note on LSTM:** underperformance attributed to the short series
  (36 months yields only ~18 usable training sequences at a 12-month
  lookback) — a known limitation of sequence models on small data, not
  a modelling error.
  - **OPD (Outpatient Department)** appears in the units reference table
    but has zero rows in `admissions.csv` — confirmed this is expected,
    not missing data: OPD activity is captured in
    `care_pathway_transitions` instead (14,646 matching rows), since
    outpatient care is tracked as ongoing monthly state rather than
    discrete admission events. Excluded from admissions-based demand
    forecasting for this reason; if OPD demand is needed later, it
    should be derived from the transitions table instead.

    ***

## Forecasting: per-unit admissions (Week 3)

- **Approach tested:** one pooled XGBoost model across all 10 admitting
  units (unit_id one-hot encoded, lag-1/lag-12/calendar-month
  features) vs. each unit's own naive 3-month trailing average.
- **Result:** pooled XGBoost beat the naive baseline on only 4 of 10
  units (DAY, DRU, FOR, GER); the naive average won on the remaining 6
  (ACF, ACM, CAU, CTU, LSR, LTW).
- **Interpretation:** with only ~18 usable training months per unit,
  the pooled model mostly learns each ward's average level via its
  unit_id flag rather than a genuine forecastable pattern; it only
  adds real value where a unit has a stronger, more stable underlying
  trend than month-to-month noise.
- **Production decision:** a per-unit "champion" approach — use
  XGBoost's forecast for DAY, DRU, FOR, and GER; use the 3-month
  trailing average for the remaining 6 units. Documented explicitly
  rather than forcing one method everywhere, since the evidence
  doesn't support it.
- **Limitation to state plainly:** demand forecasts for low-volume
  wards (LTW, CAU, CTU, GER) carry inherently higher relative
  uncertainty regardless of method, given their small monthly counts.

  ***

## Risk classifier: 90-day readmission (Week 3)

- **Label:** readmitted_90d_flag — 17.2% positive class (imbalanced).
- **Features:** admission-time and discharge-time patient/clinical
  variables only; excluded discharge_outcome (only known after the
  fact this model would predict, not a legitimate predictor).
- **Methods compared:** logistic regression (baseline, then re-run
  with feature scaling + class_weight='balanced') vs. XGBoost
  (scale_pos_weight=4.8 to address the same imbalance).
- **Result:** logistic regression outperformed XGBoost on every
  metric — ROC-AUC 0.652 vs 0.617, PR-AUC 0.303 vs 0.252, recall 0.595
  vs 0.429, F1 0.347 vs 0.316.
- **Interpretation:** with ~1,180 positive training examples, XGBoost
  likely lacks enough signal to out-learn a simpler linear
  relationship; logistic regression's lower flexibility is a better
  match for the available data size, echoing the forecasting result
  where the more complex method (LSTM) also underperformed simpler
  alternatives on limited data.
- **Production decision:** logistic regression, class-balanced,
  with features standardised at prediction time using the same
  scaler fit on training data.

  ***

## Risk classifier: 90-day readmission worklist (Week 3)

- **Worklist test:** flagged the top 10% of admissions by predicted
  risk score (threshold ≥0.677). Of those flagged, 36.0% were actually
  readmitted within 90 days, vs. a 17.2% base rate — a 2.1x lift over
  random selection.
- **Strongest predictors (by coefficient magnitude):**
  severity_on_discharge (+), medication_adherence_score (−),
  family_support_score (−), distance_km (+), prior_admissions (+) —
  a clinically coherent pattern, not a black box.
- Model and scaler saved as readmission_model.pkl / readmission_scaler.pkl.

## Risk classifier: non-attendance (Week 3)

- **Label:** did_not_attend_flag — 14.6% no-show rate.
- **Leakage confirmed and excluded:** arrival_offset_min,
  clinic_wait_min, consultation_min, and prescription_issued_flag are
  100% missing exactly when did_not_attend_flag=1 — these only exist
  because the patient attended, so including them would leak the
  answer. Excluded entirely.
- **Methods compared:** logistic regression (scaled, balanced) vs.
  XGBoost (scale_pos_weight matched to imbalance).
- **Result:** logistic regression again outperformed XGBoost on
  ROC-AUC (0.706 vs 0.691) and PR-AUC (0.386 vs 0.372); XGBoost had
  marginally higher recall (0.566 vs 0.553).
- **Cross-task pattern:** logistic regression outperformed XGBoost on
  both classifiers built this week, and LSTM underperformed simpler
  forecasting methods earlier — consistent evidence that available
  data size, not model sophistication, was the binding constraint
  throughout this phase.
- **Production decision:** logistic regression, class-balanced.

---

## Forecast endpoint: negative prediction clipping (Week 3)

- **Found:** the pooled XGBoost model predicted negative admissions
  for 3 of its 4 assigned units (DAY: -0.6, FOR: -6.7, GER: -6.1) when
  forecasting January 2026 — a physically impossible result for a
  count of patients.
- **Cause:** XGBoost regression has no inherent non-negativity
  constraint; low-volume units (all three affected units average
  under 20 admissions/month) give the model less signal to stay
  within a realistic range.
- **Fix:** predictions are clipped to a floor of 0 at the API layer,
  with a clipped_from_negative flag returned alongside the forecast so
  the frontend/report can flag when this occurred rather than hiding it.
- **Limitation to state in report:** negative-prediction clipping is a
  sign the underlying model is not perfectly suited to low-volume
  count data; a Poisson or negative-binomial regression objective
  would be a more principled fix if revisited beyond this capstone's
  scope.

  ***

## Forecast validation gap: champion selection never tested a January (Week 3)

- **Found:** in production, forecasting January 2026 caused 3 of the 4
  XGBoost-assigned units (DAY, FOR, GER) to predict negative
  admissions before clipping (-0.6, -6.7, -6.1).
- **Root cause identified:** the train/test split held out July-December
  2025 as the test period, which never included a January. The
  "XGBoost beats naive" result for these 4 units was therefore never
  actually validated against a January prediction -- it's an
  out-of-sample month-type for the champion decision itself.
- **Mitigation applied:** predictions are floored at 0 in the API, with
  a clipped_from_negative flag so this is visible, not hidden.
- **Limitation to state plainly:** with only ~2 historical Januaries
  per unit in the training window, any January-specific pattern the
  model learned is built on very thin evidence. Recommend treating
  XGBoost forecasts for calendar months underrepresented in the
  training/test data with added caution, or falling back to the naive
  method specifically for January until more years of data exist.

---

## Patient segmentation (Week 3) — final

- **Method progression:** K-means (biased toward one-hot categorical
  columns, near-zero ARI/NMI) → K-Prototypes unscaled (biased toward
  large-magnitude numeric columns, distance_km alone, near-zero
  ARI/NMI) → K-Prototypes with scaled numerics and travel_fare_ngn
  dropped (redundant with distance_km) — final approach.
- **k selection:** cost-reduction-per-step analysis showed a
  consistent halving of returns around k=5 (6.6% → 4.1%), corroborated
  independently by the unscaled run's sharper elbow at the same k.
- **Result:** ARI 0.2533, NMI 0.3187 against segment_label — a
  genuine, meaningful improvement over both prior attempts (~0.0003 /
  0.001).
- **Cluster interpretation:** 4 clinically distinct groups (stable
  older/well-supported; younger/substance-use burden; new/low-risk;
  chronic long-term high-utilizer) plus 1 geographically-isolated
  group defined by distance rather than clinical profile.
- **Key finding:** cluster 3 (chronic, low-support, high prior
  admissions) is a strong candidate explanation for the board's
  stated "repeat spend without resolution" concern, complementing the
  35–43% Lost-to-follow-up finding from Week 2.

  ***

## Model 2 input decision: census-based demand, not Little's Law (Week 4)

- **Found:** Little's Law (forecast admission rate x avg length of stay)
  produced clearly wrong results for LTW (1.1 beds needed vs 50 actual)
  and was suspect for DAY, since both units don't have a conventional
  admission-driven occupancy pattern.
- **Found:** daily_census.csv provides directly observed occupancy,
  including beds_short and patients_waiting_for_bed -- ground truth,
  not a derived estimate.
- **Key finding:** DRU and LSR are running at 121% and 118% average
  occupancy respectively over the last 6 months (via overflow, not
  nominal capacity), with ~17.5 and ~9.8 beds short per day on average
  -- the clearest, most severe capacity gap in the dataset.
- **Found:** long_stay_residents.csv (39 current) undercounts LTW's
  true occupancy (49.17 per the census) -- the registry appears to
  track only formally grant-funded residents, not every occupant.
- **Decision:** Model 2's demand-satisfaction constraint uses
  avg_occupied_beds from the last 6 months of daily_census per unit,
  not a forecast-derived Little's Law estimate. The Week 3 forecast is
  reserved for scenario analysis (demand growth, etc.) later.

  ***

## Model 2: core optimisation solved (Week 4)

- **Formulation:** minimise total annual cost (beds + nurses + doctors,
  by unit) subject to: meeting each unit's 6-month average observed
  occupancy (daily_census.csv, not a forecast-derived estimate --
  see prior entry), not exceeding total physical bed stock (372),
  maintaining each unit's own currently-observed nurse/doctor-per-bed
  ratio as a floor, and staying within the 2026 allocated budget.
- **Status:** Optimal (PuLP / CBC solver).
- **Result:** total annual cost NGN 4.70B vs. NGN 5.35B at current bed
  counts -- a ~12.2% (NGN 652M/year) reduction.
- **Key finding:** DRU (+9 beds, +22.5%) and LSR (+8 beds, +17.8%) are
  the only units recommended to grow -- both independently confirmed
  as over-capacity (121%, 118% occupancy) by the census data before
  the model was even built. All other units shrink, most sharply DAY
  (-80%) and CTU (-88.9%).
- **Limitation to state plainly:** the model optimises against average
  demand only; it does not yet account for demand variability or
  surge capacity, so large recommended cuts (DAY, CTU) should be
  stress-tested against the Week 5 scenarios before being treated as
  final, not adopted directly from this single solve.

  ***

## Model 2 limitation: locked_flag not respected in the core solve (Week 4)

- **Found:** Model 2's optimisation treats all 372 beds as freely
  reallocatable between any unit. beds.csv's locked_flag shows 257 of
  372 beds (69%) are 100%-locked at the unit level (ACF, ACM, CAU,
  CTU, FOR, GER, LTW) -- likely reflecting clinical/structural
  fixedness (e.g. a forensic secure bed cannot become a day-hospital
  bed). Only DAY, DRU, and LSR (115 beds) are unlocked.
- **Implication:** Model 2's recommended cuts to locked units are
  probably not physically implementable. The only genuinely
  actionable recommendation from that solve is the reallocation
  within the flexible pool: DAY shrinks to grow DRU and LSR.
- **Resolution:** the compulsory bed-reallocation task (below) is
  scoped correctly to the flexible pool only, respecting locked_flag
  as a hard constraint. Model 2's full-hospital recommendation should
  be presented to the board with this caveat explicit, or re-run with
  locked beds fixed at their current per-unit count as a stated
  follow-up.

  ***

## Compulsory bed reallocation: LP -> MIP with sensitivity (Week 4)

- **Scope correction:** reallocation restricted to the 115-bed
  flexible pool (DAY, DRU, LSR) only, respecting locked_flag as a
  hard constraint -- see prior entry on why the other 7 units'
  beds are not genuinely reallocatable.
- **LP result:** DAY 13.00, DRU 49.00, LSR 53.00 -- already
  integer-valued, so the MIP stage confirmed the identical solution
  with no rounding compromise required.
- **Shadow prices (sensitivity analysis):**
  - DRU: NGN 9,295,029/year per additional bed of demand -- the
    highest-value unit for new bed investment
  - LSR: NGN 5,205,306/year per additional bed of demand
  - DAY: NGN 0 -- non-binding, DAY holds 7 beds of slack above its
    minimum requirement within this reallocation
  - Pool size itself: NGN 6,977,583/year saved per additional bed
    added to the flexible pool -- a quantified case for unlocking
    capacity beyond simple reshuffling
- **Recommendation:** move 17 beds from DAY to DRU (+9) and LSR (+8)
  within the existing flexible pool, at zero net capital cost since
  total flexible beds are conserved (115 in, 115 out).

  ***

## Roster sensitivity: variable-reuse bug and corrected result (Week 4)

- **Bug found:** an initial night-shift-cap sensitivity test reused
  the same PuLP variable objects across two separate LpProblem
  instances. Since a PuLP variable's .value() reflects whichever
  problem was solved most recently, solving the second (relaxed)
  model silently overwrote the first model's reported objective value
  too, producing a false "0.0% improvement" that was actually
  comparing the relaxed model against itself.
- **Fix:** rebuilt both scenarios as fully independent models with
  uniquely-suffixed variable names, eliminating any shared state.
- **Corrected result:** relaxing the night-shift cap by 2/person/month
  reduces total penalised shortfall by only 0.6% (2542.94 -> 2528.33).
- **Conclusion:** the night-shift cap is not the binding constraint on
  the nurse shortfall, despite 70% of staff sitting at their current
  cap -- that statistic reflects understaffing, not a policy
  bottleneck. The 1,733 unfilled nurse-shifts over 28 days are a
  genuine headcount shortage, concentrated in ACF (409 unfilled) and
  FOR (402 unfilled), requiring hiring or cross-unit transfer rather
  than a scheduling policy change.

  ***

## Compulsory budget goal-programming model (Week 4)

- **Framing:** 2026 budget already has amount_allocated_ngn populated
  (5.67B against 7.72B requested, a 26.5% structural shortfall) --
  this is a re-allocation/prioritisation problem, not a from-scratch
  budgeting problem.
- **Method:** lexicographic (tiered) goal programming across 88
  unit/cost_category goals, 3 tiers, weighted by
  strategic_priority_weight, solved as 3 sequential stages (each
  locking in the prior tier's achieved shortfall before optimising
  the next).
- **Bug/limitation found:** a naive strict-lock implementation
  produced a 0.0%-funded Personnel line at CAU -- mathematically
  optimal but operationally impossible (a ward cannot run with zero
  staff budget). Root cause: strict tiering gives the solver no
  incentive to protect anything outside the tier currently being
  optimised.
- **Fix:** added a hard floor (70% of prior-year actual spend) on all
  Personnel lines regardless of tier, and loosened the tier-lock
  tolerance to 2% to allow limited cross-tier trade-offs. Result:
  Personnel funded at 100% for 9 of 11 units, 48-49% for the
  remaining 2 (both hit their floor exactly); tier 3 average funding
  rose from 0.0% to 5.5%.
- **Headline finding:** the 7.72B requested vs. 5.67B available gap is
  structural, not a modelling failure -- roughly 26.5% of total
  hospital budget requests cannot be met regardless of allocation
  method, and tier 3 (lowest strategic priority) categories absorb
  nearly all of that shortfall by design.

---

## Simulation build and validation (Week 5)

- **Design decisions made before building:**
  - Monthly clock, matching the Week 2 transition matrix (the brief
    treats a monthly-matrix-fed-daily-simulation as an error).
  - States are pathway stages, not beds. A per-state bed fraction phi
    was required after reconciling the chain's cohort headcount
    against census beds in the same period: acute ratio ~5.4x,
    rehab ~2.4x, day hospital ~31x -- one scale factor could not
    reconcile all three, so phi is fitted per state.
  - phi calibrated against steady-state analytic headcount
    (arrivals x start_mix x N) in the census window 2023-04 to
    2025-12, excluding the Jan-Feb 2023 warm-up (hospital starts
    empty in hospital_daily_series.csv, occupancy ~142 beds in month
    1 rising to ~275-300 by month 3).
  - Overcrowding-to-dropout mechanism included but OFF by default:
    the transition matrix was estimated while rehab already ran at
    118-121% occupancy, so P already contains whatever effect
    overcrowding has on outcomes. There is no data to separate bed
    availability from outcomes, so beds drive pressure
    (occupancy/overflow/wait/cost) only; outcomes move solely through
    explicit transition changes.

- **Self-tested against a synthetic matrix first** (6 checks: analytic
  vs simulated headcount, matrix B vs simulated absorption mix,
  row-stochasticity preservation under modify_P/scale_discharges,
  exact phi recovery, capacity-extreme sanity, replication interval
  sanity) -- all passed, confirming the engine's core arithmetic
  before validating against real data.

- **Validation A (the real test):** P trained on 80% of patients,
  predicted absorption mix compared against the held-out 20%'s actual
  outcomes. All 5 outcomes matched within 0.008 (Deceased) down to
  0.003 (Long stay). This is a genuine out-of-sample test, distinct
  from the earlier full-cohort check (which only confirmed the
  maximum-likelihood arithmetic was correct, since a matrix fit on
  the whole cohort must reproduce that cohort's own outcomes).

- **Validation B:** simulated occupancy vs census, informational only
  -- phi was calibrated to hit this exactly, so it cannot fail by
  construction. Reported as such rather than presented as independent
  evidence.

- **Validation C (independent, uncalibrated):** simulated overflow
  compared against crisis_unit_hourly.referred_out_no_bed, a signal
  the calibration never touched.
  - Rehab: 13.4 beds overflow, 115.7% occupancy, ~9 day wait --
    matches the direction and severity of the Week 4 census finding
    (DRU/LSR at 118-121% occupancy).
  - Acute: 0.0 beds overflow simulated, vs ~31 people/month turned
    away for no bed in the real crisis-unit data -- a real mismatch,
    not a rounding difference.
  - Root cause: the monthly clock only sees a month's AVERAGE
    occupancy (72% for acute), which cannot represent short daily
    bursts of demand that fill beds temporarily even when the
    monthly average has headroom. This is the unit/time-base mismatch
    the brief warns about, surfaced by the one validation check that
    wasn't calibrated to agree.
  - Not fixed in this pass: a daily sub-step for capacity-checking
    only (transition matrix stays monthly) would close this gap.
    Stated here as an explicit, scoped limitation rather than forced
    through under time pressure.
  - Trend note: this simulation's full-period (33-month) average
    rehab overflow (13.4 beds) is noticeably lower than Week 4's
    recent-6-month-only census figure (~27.3 beds combined across
    DRU/LSR), suggesting the overcrowding has worsened over time
    rather than been constant -- worth a line in the report.
