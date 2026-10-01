"""
WardWise patient-journey simulation (Week 5).

A patient-level, discrete-event simulation on a MONTHLY clock. The clock
is monthly because the transition matrix estimated in Week 2 is monthly:
the brief is explicit that a monthly matrix feeding a daily simulation is
an error, so every rate, count and cost here is per month.

Events, processed each month:
  1. arrivals   - Poisson(rate x seasonal index x demand multiplier x event factor)
  2. transitions - every active patient draws its next state from row P[state]
  3. absorption  - patients reaching an absorbing state are frozen there
  4. relapse     - an entry into 'Community relapse' from another state, i.e.
                   relapse feeding back to the front door via the chain

Beds vs the chain.  The chain's states are pathway STAGES, not beds. Only a
fraction phi of the patient-months in a bed-based stage occupy a bed in the
census (calibrated against steady-state census occupancy, see
calibrate_phi). Beds therefore drive pressure - occupancy, overflow, waiting,
cost - and never move a transition by themselves. Outcomes change only
through explicit transition changes (modify_P) or through the optional
overcrowd_dropout mechanism, which is OFF by default because the data cannot
separate bed availability from outcomes (P was estimated while rehab ran at
~120% occupancy).

Waiting time is a Little's-law proxy: overflow patients / entry rate. It is
not directly observed.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

STATES = [
    "Acute inpatient", "Rehabilitation inpatient", "Day hospital",
    "Outpatient follow-up", "Community relapse",
    "Deceased", "Long stay institutional care", "Lost to follow-up",
    "Recovered and discharged from service", "Transferred to another facility",
]
N_T = 5                      # transient states are STATES[0:5]
BED_STATES = (0, 1, 2)       # acute, rehab, day hospital
ABSORBING = STATES[N_T:]
DAYS_PER_MONTH = 30.4
LTFU = 7                     # index of 'Lost to follow-up'
DISCHARGE_COLS = (8, 9)      # recovered, transferred


@dataclass
class SimConfig:
    P: np.ndarray                     # 10x10 monthly transition matrix
    start_probs: np.ndarray           # length 5, distribution over transient start states
    arrivals_per_month: float
    seasonal_index: np.ndarray        # length 12, mean 1
    phi: dict                         # {0: frac, 1: frac, 2: frac}  beds per chain patient
    capacity: dict                    # {0: beds, 1: beds, 2: beds}
    bed_cost_per_day: dict            # {0: ngn, 1: ngn, 2: ngn}
    community_cost_per_month: dict    # {3: ngn, 4: ngn}   per patient-month
    start_month: int = 0              # calendar month (0=Jan) of simulation month 0
    horizon: int = 108
    warmup: int = 12
    entry_window: int = 48            # months of entrants tracked as the outcome cohort
    demand_multiplier: float = 1.0
    events: dict = field(default_factory=dict)   # {sim_month: (arrival_factor, discharge_factor)}
    overcrowd_dropout: float = 0.0    # extra monthly LTFU prob per unit of (occupancy_rate - 1); OFF by default


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def fundamental(P):
    Q = P[:N_T, :N_T]
    return np.linalg.inv(np.eye(N_T) - Q)


def absorption_matrix(P):
    """B = N R, shape (5 transient, 5 absorbing)."""
    return fundamental(P) @ P[:N_T, N_T:]


def analytic_headcount(cfg):
    """Steady-state expected patients per transient state: lambda * start_probs @ N."""
    lam = cfg.arrivals_per_month * cfg.demand_multiplier
    return lam * (cfg.start_probs @ fundamental(cfg.P))


def calibrate_phi(cfg, census_means):
    """phi_g = mean census occupied beds / steady-state chain headcount in that stage.
    Anchors baseline occupancy to the census by construction (a calibration, not a test)."""
    base = replace(cfg, demand_multiplier=1.0)
    head = analytic_headcount(base)
    return replace(cfg, phi={g: census_means[g] / head[g] for g in BED_STATES})


def modify_P(P, src, dst, factor, redirect=None):
    """Scale P[src, dst] by `factor`; the mass removed (or added) is taken from /
    given to P[src, redirect] (default: the patient stays where they are)."""
    P2 = P.copy()
    redirect = src if redirect is None else redirect
    delta = P2[src, dst] * (factor - 1.0)
    P2[src, dst] += delta
    P2[src, redirect] -= delta
    if (P2[src] < -1e-12).any():
        raise ValueError("modify_P produced a negative probability")
    return P2


def scale_discharges(P, factor):
    """Scale exits to 'Recovered' / 'Transferred'; removed mass stays in place."""
    P2 = P.copy()
    for j in DISCHARGE_COLS:
        removed = P2[:N_T, j] * (1.0 - factor)
        P2[:N_T, j] -= removed
        P2[np.arange(N_T), np.arange(N_T)] += removed
    return P2


# ----------------------------------------------------------------------------
# one replication
# ----------------------------------------------------------------------------
def simulate(cfg, seed=0):
    rng = np.random.default_rng(seed)
    T = cfg.horizon
    cdf_cache = {}

    def cdf_for(discharge_factor):
        if discharge_factor not in cdf_cache:
            P = cfg.P if discharge_factor == 1.0 else scale_discharges(cfg.P, discharge_factor)
            c = np.cumsum(P, axis=1)
            c[:, -1] = 1.0
            cdf_cache[discharge_factor] = c
        return cdf_cache[discharge_factor]

    counts = np.zeros((T, 10), dtype=np.int64)
    entries = np.zeros((T, 3), dtype=np.int64)
    arrivals = np.zeros(T, dtype=np.int64)
    relapse_events = np.zeros(T, dtype=np.int64)

    state = np.empty(0, dtype=np.int8)
    entry_t = np.empty(0, dtype=np.int32)
    absorb_t = np.empty(0, dtype=np.int32)
    n_relapse = np.empty(0, dtype=np.int32)

    for t in range(T):
        arr_f, dis_f = cfg.events.get(t, (1.0, 1.0))
        lam = (cfg.arrivals_per_month * cfg.seasonal_index[(cfg.start_month + t) % 12]
               * cfg.demand_multiplier * arr_f)
        n_new = rng.poisson(lam)
        arrivals[t] = n_new
        if n_new:
            s_new = rng.choice(N_T, size=n_new, p=cfg.start_probs).astype(np.int8)
            state = np.concatenate([state, s_new])
            entry_t = np.concatenate([entry_t, np.full(n_new, t, dtype=np.int32)])
            absorb_t = np.concatenate([absorb_t, np.full(n_new, -1, dtype=np.int32)])
            n_relapse = np.concatenate([n_relapse, np.zeros(n_new, dtype=np.int32)])
            for g in BED_STATES:
                entries[t, g] += int(np.sum(s_new == g))

        counts[t] = np.bincount(state, minlength=10)

        idx = np.flatnonzero(state < N_T)
        if idx.size:
            cdf = cdf_for(dis_f)
            rows = cdf[state[idx]].copy()
            if cfg.overcrowd_dropout > 0.0:
                # optional, off by default: extra LTFU risk for patients in an over-capacity bed stage
                extra = np.zeros(idx.size)
                for g in BED_STATES:
                    occ = cfg.phi[g] * counts[t, g] / cfg.capacity[g]
                    if occ > 1.0:
                        extra[state[idx] == g] = cfg.overcrowd_dropout * (occ - 1.0)
                # shift the LTFU mass: probability `extra` is diverted to LTFU
                u = rng.random(idx.size)
                to_ltfu = u < extra
            else:
                to_ltfu = np.zeros(idx.size, dtype=bool)
            u2 = rng.random(idx.size)
            nxt = (u2[:, None] > rows).sum(axis=1).astype(np.int8)
            nxt = np.minimum(nxt, 9)
            nxt[to_ltfu] = LTFU
            cur = state[idx]
            moved = nxt != cur
            for g in BED_STATES:
                entries[t, g] += int(np.sum(moved & (nxt == g)))
            rel = moved & (nxt == 4)
            relapse_events[t] = int(rel.sum())
            n_relapse[idx[rel]] += 1
            newly_abs = nxt >= N_T
            absorb_t[idx[newly_abs]] = t + 1
            state[idx] = nxt

    return dict(counts=counts, entries=entries, arrivals=arrivals,
                relapse_events=relapse_events, state=state, entry_t=entry_t,
                absorb_t=absorb_t, n_relapse=n_relapse)


def summarise(cfg, res):
    w = cfg.warmup
    counts = res["counts"][w:]
    out = {}
    bed_cost = 0.0
    for g in BED_STATES:
        demand = cfg.phi[g] * counts[:, g]
        cap = cfg.capacity[g]
        overflow = np.maximum(demand - cap, 0.0)
        entry_rate = res["entries"][w:, g].mean()
        out[f"beds_demand_{g}"] = demand.mean()
        out[f"occupancy_rate_{g}"] = demand.mean() / cap
        out[f"overflow_beds_{g}"] = overflow.mean()
        out[f"wait_days_{g}"] = (overflow.mean() / cfg.phi[g]) / entry_rate * DAYS_PER_MONTH if entry_rate > 0 else 0.0
        bed_cost += demand.mean() * cfg.bed_cost_per_day[g] * DAYS_PER_MONTH

    cohort = (res["entry_t"] >= w) & (res["entry_t"] < w + cfg.entry_window)
    final = res["state"][cohort]
    absorbed = final >= N_T
    props = np.bincount(final[absorbed] - N_T, minlength=5) / max(absorbed.sum(), 1)
    for name, p in zip(ABSORBING, props):
        out[f"abs_{name}"] = p
    out["censored_share"] = 1.0 - absorbed.mean()
    out["relapses_per_patient"] = res["n_relapse"][cohort].mean()
    dur = (res["absorb_t"][cohort] - res["entry_t"][cohort])[absorbed]
    out["months_in_system"] = dur.mean()

    comm_cost = float(np.mean(counts[:, 3] * cfg.community_cost_per_month[3]
                              + counts[:, 4] * cfg.community_cost_per_month[4]))
    arr = res["arrivals"][w:].mean()
    out["monthly_cost_ngn"] = bed_cost + comm_cost
    out["bed_cost_share"] = bed_cost / (bed_cost + comm_cost) if (bed_cost + comm_cost) else 0.0
    out["cost_per_new_patient_ngn"] = (bed_cost + comm_cost) / arr if arr > 0 else 0.0
    out["arrivals_per_month"] = arr
    out["monthly_relapse_events"] = res["relapse_events"][w:].mean()
    return out


def run_replications(cfg, n_reps=100, seed0=0):
    """Returns a DataFrame with mean and 95% interval (2.5-97.5 percentile) per metric."""
    rows = [summarise(cfg, simulate(cfg, seed0 + r)) for r in range(n_reps)]
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "mean": df.mean(),
        "lo95": df.quantile(0.025),
        "hi95": df.quantile(0.975),
    })
