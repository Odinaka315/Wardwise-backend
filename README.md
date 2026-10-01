# FNPH Yaba — Model 2 starter (bed & staffing configuration)

One-command starter for the capstone: Postgres + Redis + FastAPI + Celery,
already wired together, plus a working absorbing-chain calculation as the
first real piece of logic.

## Run it

1. Copy `.env.example` to `.env` (defaults work as-is for local dev):
   ```
   cp .env.example .env
   ```
2. Put your dataset dump at `db/init/01-fnph-dataset.sql`
   (see `db/init/README.txt`).
3. Start everything:
   ```
   docker compose up --build
   ```
4. Check it's alive:
   - API health + DB connectivity: http://localhost:8000/health
   - Interactive API docs: http://localhost:8000/docs
   - Adminer (browse the DB in a browser): http://localhost:8081
     — System: PostgreSQL, Server: `db`, Username: `fnph`, Password: `fnph`,
     Database: `fnph_yaba`

## What's already here

- `app/db.py` — SQLAlchemy connection, reused by every endpoint
- `app/celery_app.py` — background job queue config. Long solves (the
  Model 2 optimisation, simulations) get written as Celery tasks in
  `app/services/`, never run inside a request.
- `app/services/pathway.py` — builds the transition matrix from
  `care_pathway_transitions`, splits it into Q and R, computes the
  fundamental matrix N and the absorption matrix B = NR. Runnable
  standalone: `docker compose exec api python -m app.services.pathway`
- `app/routers/pathway.py` — exposes that as `GET /api/v1/pathway/summary`

## Known placeholder to fix first

`app/services/pathway.py` assumes `care_pathway_transitions` has columns
`patient_id, month, from_state, to_state` with specific state name strings
(see `TRANSIENT_STATES` / `ABSORBING_STATES` at the top of the file). Open
the real table in Adminer once your dump is loaded and correct these to
match — this is deliberately the first thing that will need adjusting,
not a bug.

## Next things to build, in order

1. Fix the column/state-name assumptions above against the real schema.
2. Data audit + `CLEANING_LOG.md` — the three admissions columns with
   missing values and the ~20 inflated length-of-stay values the brief
   mentions.
3. Descriptive baseline endpoint (occupancy by ward, nominal vs
   effective capacity) — this is the Baseline screen.
4. Demand forecasting per unit — feeds Model 2 directly (it's one of
   its stated inputs).
5. The Model 2 formulation itself (bed & staffing configuration) as a
   Celery task using PuLP or OR-Tools, once the transition matrix and
   forecasts exist to parameterise it.

## Notes

- No CSVs in this running system — the database is the single source
  of truth, per the brief. Use the CSV zip only inside your model
  *training* notebooks (kept outside this repo, or in a `/notebooks`
  folder you add), never as something the API reads at runtime.
- `RANDOM_SEED` is in `.env` so every trained artifact and every solve
  can record which seed produced it, per the reproducibility
  requirement.
