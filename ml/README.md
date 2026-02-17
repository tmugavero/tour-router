# Tour Recommendation ML Pipeline

Cross-artist collaborative filtering model for live music tour planning.

## Quick Start

```bash
cd ml
pip install -r requirements.txt

# Train on your settlement CSVs and export for the frontend
python train_and_export.py --csv-dir ./data/settlements --output ./export.json
```

## Settlement CSV Format

Each CSV should contain one artist's show settlement data with these columns:

| Column | Description |
|--------|-------------|
| `date` | Show date (YYYY-MM-DD) |
| `city` | City name |
| `state` | State abbreviation |
| `country` | Country (default: United States) |
| `venue` | Venue name |
| `capacity` | Venue capacity |
| `tickets_sold` | Tickets sold |
| `guarantee` | Artist guarantee ($) |
| `artist_net` | Artist net revenue ($) |
| `avg_ticket_price` | Average ticket price ($) |
| `show_type` | "headline", "support", "festival" |
| `artist` | Artist name |

## Project Structure

```
ml/
├── tour_recommender/
│   ├── __init__.py        # Package init, top-level initialize()
│   ├── data.py            # CSV parsing, feature engineering, FeatureStore
│   ├── model.py           # Three-head ML model, route optimization
│   └── tool.py            # Tool interface for LangGraph orchestrator
├── train_and_export.py    # Train model → export JSON for frontend
├── demo.py                # Demo script exercising all three input modes
└── requirements.txt
```

## Three Input Modes

1. **Known artist** — has settlement data, full historical features
2. **Unknown artist + streaming** — Spotify listeners mapped to capacity/guarantee tier
3. **Manual profile** — pass raw features directly

## Updating the Frontend

After running `train_and_export.py`, the output JSON contains:
- `markets`: Updated market profiles → paste into `MARKETS` in `TourApp.jsx`
- `scenarios`: Pre-computed artist results → paste into `PRECOMPUTED` in `TourApp.jsx`

The more settlement CSVs you add, the better the cross-artist model gets.
