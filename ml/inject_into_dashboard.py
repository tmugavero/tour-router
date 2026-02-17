"""
Inject export.json data into TourApp.jsx with enhanced dashboard.
Generates the complete TourApp.jsx with Profitability and Timing tabs.
"""
import json
import sys
import os

def main():
    export_path = os.path.join(os.path.dirname(__file__), 'export.json')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'TourApp.jsx')

    with open(export_path, 'r') as f:
        data = json.load(f)

    markets = data['markets']
    scenarios = data['scenarios']
    seasonality = data.get('seasonality', {})
    metadata = data['model_metadata']

    # Build COORDS from model
    sys.path.insert(0, os.path.dirname(__file__))
    from tour_recommender.model import MARKET_COORDS
    coords = {}
    for m in markets:
        key = m['m']
        if key in MARKET_COORDS:
            lat, lon = MARKET_COORDS[key]
            coords[key] = [round(lat, 2), round(lon, 2)]

    # Build DEMOS
    phase_labels = {'arena': 'Arena', 'theater': 'Theater', 'club_to_theater': 'Club->Theater', 'club': 'Club', 'emerging': 'Emerging'}
    icons = {'arena': '\U0001f3df\ufe0f', 'theater': '\U0001f3ad', 'club_to_theater': '\U0001f4c8', 'club': '\U0001f3b5', 'emerging': '\U0001f331'}
    demos = []
    for key, sc in sorted(scenarios.items()):
        phase = sc['artist_profile']['growth_phase']
        demos.append({
            'key': key,
            'label': sc['artist'],
            'sub': f"{phase_labels.get(phase, phase)} \u00b7 Settlement",
            'icon': icons.get(phase, '\U0001f3b5'),
        })

    # Build JSON strings
    markets_json = json.dumps(markets, separators=(',', ':'))
    coords_json = json.dumps(coords, separators=(',', ':'))
    precomputed_json = json.dumps(scenarios, separators=(',', ':'))
    demos_json = json.dumps(demos, separators=(',', ':'), ensure_ascii=False)
    seasonality_json = json.dumps(seasonality, separators=(',', ':'))

    n_shows = metadata['n_training_shows']
    n_mkts = metadata['n_markets']
    n_feats = metadata.get('n_features', 44)
    fill_auc = metadata.get('fill_auc', 0)
    revenue_r2 = metadata.get('revenue_r2', 0)
    guarantee_r2 = metadata.get('guarantee_r2', 0)

    # Write template with replaced markers
    jsx_template = open(os.path.join(os.path.dirname(__file__), 'dashboard_template.jsx'), 'r', encoding='utf-8').read()
    jsx = jsx_template
    jsx = jsx.replace('__MARKETS_JSON__', markets_json)
    jsx = jsx.replace('__COORDS_JSON__', coords_json)
    jsx = jsx.replace('__PRECOMPUTED_JSON__', precomputed_json)
    jsx = jsx.replace('__DEMOS_JSON__', demos_json)
    jsx = jsx.replace('__SEASONALITY_JSON__', seasonality_json)
    jsx = jsx.replace('__N_SHOWS__', str(n_shows))
    jsx = jsx.replace('__N_MARKETS__', str(n_mkts))
    jsx = jsx.replace('__N_FEATURES__', str(n_feats))
    jsx = jsx.replace('__FILL_AUC__', str(fill_auc))
    jsx = jsx.replace('__REVENUE_R2__', str(revenue_r2))
    jsx = jsx.replace('__GUARANTEE_R2__', str(guarantee_r2))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(jsx)

    print(f"Generated TourApp.jsx ({len(jsx):,} chars)")
    print(f"  {len(markets)} markets, {len(scenarios)} scenarios, {len(seasonality)} seasonality profiles")
    print(f"  {len(demos)} demo artists")


if __name__ == '__main__':
    main()
