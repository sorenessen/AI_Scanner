# Runtime Configuration (v0.3.3)

These settings are persisted server-side via `/config` and reflected in the UI.

## Defaults (Balanced)

| Key                  | Default | Purpose |
|----------------------|---------|---------|
| `mode`               | `Balanced` | Baseline tuning (vs. `Strict`, `Academic`). |
| `min_tokens_strong`  | `180`   | Minimum tokens before allowing a “strong” decision. |
| `short_cap`          | `true`  | Apply confidence ceiling to short excerpts. |
| `max_conf_short`     | `0.35`  | Ceiling when text is short. |
| `non_en_cap`         | `0.15`  | Ceiling when text appears non-English. |
| `en_thresh`          | `0.70`  | Probability threshold for English detection. |
| `max_conf_unstable`  | `0.35`  | Ceiling when internal instability is detected. |
| `abstain_low`        | `0.35`  | Lower bound of inconclusive band. |
| `abstain_high`       | `0.65`  | Upper bound of inconclusive band. |
| `use_ensemble`       | `false` | If secondary model is loaded, combine judgments. |

> **Operator note:** PD fingerprints are optional. To enable PD overlap hints:
> put centroid JSONs into `./pd_fingerprints/` and restart the service.
> `/version` should report `fingerprint_centroids: N`.

## Modes

- **Balanced**: General purpose.
- **Strict**: More conservative to call *human*; shifts toward abstaining.
- **Academic**: Tuned for longer formal prose; higher min‐tokens recommended.

## API Contract

- `GET /config` → `{ settings: { … } }`
- `POST /config` → accepts the keys above. UI re-loads version after save.
- `GET /version` → should include:  
  `version, model, device, dtype, ensemble, fingerprint_centroids`.

## Change Log (config-relevant)

- **v0.3.3**
  - Added PD badge and PD note in Explain when centroids missing.
  - Footer hint for `./pd_fingerprints/`.
  - No behavioral change to detection/calibration defaults.
