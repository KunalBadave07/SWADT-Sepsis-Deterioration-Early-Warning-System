# Sprint 5, Milestone 1.5 — REQUIRED FIX: Risk Table Must Use Real Streamed Data
### Close this before touching any visual polish.

## The problem, precisely
Your bed grid shows real admitted patients (from `census_runner.py`, streaming actual
PhysioNet files through the real pipeline into Feast/Redis). Your risk table calls
`predict(patient_id, random_vitals())` — fabricated numbers with no relationship to
that patient's real data. The two halves of your dashboard are lying to each other.

## The fix
Pull each occupied bed's ACTUAL latest features from Feast — the same online store
your Sprint 1 pipeline has already been populating this whole time — instead of
generating fake ones.

In `dashboard/app.py`, add near the top (after your existing imports):
```python
from feast import FeatureStore

feast_store = FeatureStore(repo_path="pipeline/sepsis_feast")

FEATURE_REFS = [
    "patient_vitals:heart_rate", "patient_vitals:resp_rate", "patient_vitals:sbp",
    "patient_vitals:map_bp", "patient_vitals:temp_c", "patient_vitals:spo2",
    "patient_vitals:hr_rolling_mean", "patient_vitals:hr_rolling_std",
    "patient_vitals:map_rolling_mean", "patient_vitals:map_rolling_std",
    "patient_vitals:shock_index",
]


def get_live_vitals(patient_id: str) -> dict | None:
    """
    Pulls this patient's REAL, currently-streamed rolling features from
    Feast — the same data the census bed grid's patient is actually
    generating right now. Returns None if the patient was just admitted
    and hasn't had a feature pushed yet (still "warming up").
    """
    result = feast_store.get_online_features(
        features=FEATURE_REFS, entity_rows=[{"patient_id": patient_id}]
    ).to_dict()

    if result.get("heart_rate", [None])[0] is None:
        return None  # no real data yet for this patient — don't fake it

    return {
        key.split(":")[-1]: values[0]
        for key, values in result.items()
        if key != "patient_id"
    }
```

Now replace the old `random_vitals()`-driven loop that builds `st.session_state.results`
with this — using the REAL occupied beds from the census, and REAL features per patient:
```python
census = get_census()
occupied_beds = [b for b in census if b["status"] == "occupied"]

if st.button("🔄 Refresh risk scores"):
    st.session_state.results = {}

for bed in occupied_beds:
    pid = bed["patient_id"]
    if pid not in st.session_state.results:
        live_vitals = get_live_vitals(pid)
        if live_vitals is None:
            continue  # patient admitted but no features streamed yet — skip, don't fake
        st.session_state.results[pid] = predict(pid, live_vitals)
```

**And handle the "warming up" case explicitly in the UI** — don't let a missing patient
silently disappear without explanation:
```python
skipped = [b["patient_id"] for b in occupied_beds if b["patient_id"] not in st.session_state.results]
if skipped:
    st.caption(f"⏳ Still gathering initial readings for: {', '.join(skipped)}")
```

## Verify the fix actually worked, don't just trust it
1. Run the full stack (all four terminals from Milestone 1, Step 7).
2. Let the census run for a couple minutes so patients accumulate real streamed features.
3. Refresh the dashboard. **Look at the risk tier distribution now.** You should see a genuine
   mix — some `stable`, some `at_risk`, maybe one or two `deteriorating` — not a wall of red.
4. Pick one patient, note their `patient_id`, and cross-check: does their SHAP explanation's
   top feature (e.g., `map_bp`, `hr_rolling_mean`) plausibly match what a real vitals trend for
   that patient would look like? You can spot-check this against the raw `.psv` file for that
   same patient ID in `data/raw/training/` if you want real confidence, not just a vibe check.

**If you STILL see mostly at_risk/deteriorating after this fix**, that's a different, more
important finding: it would mean the model itself is somewhat miscalibrated toward
over-predicting risk (plausible given the aggressive `scale_pos_weight=18.28` and the
0.138 Brier score already logged in Sprint 2) — worth documenting honestly in
`PRODUCTION_NOTES.md` as a known calibration issue for future work (e.g., Platt scaling
recalibration), rather than something to quietly hope nobody notices.

## Definition of Done for this fix
- [ ] Risk table pulls real Feast features per real census patient, not `random_vitals()`
- [ ] "Still gathering initial readings" state handled visibly, not silently dropped
- [ ] Risk tier distribution across real patients shows genuine variety — not uniformly alarming
- [ ] If it's still mostly high-risk after the fix, that's documented as a real calibration
      finding, not hidden

Once this is genuinely fixed and verified, Milestone 2 (the real bed-grid visual design —
restrained color coding, purposeful animation) is next, and it'll actually mean something
once the numbers underneath it are real.
