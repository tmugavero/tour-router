"""
Train the cross-artist ML model and export results for the Next.js frontend.

Usage:
    python train_and_export.py --csv-dir ./data/settlements --output ../app/data/scenarios.json

This script:
  1. Loads all settlement CSVs from the specified directory
  2. Builds the FeatureStore (artist profiles, market profiles, training matrix)
  3. Trains the three-head ML model (fill, revenue, capacity)
  4. Runs predictions for all known artists
  5. Exports market data + pre-computed scenarios as JSON for the frontend
"""

import argparse
import json
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tour_recommender import initialize, recommend_tour_markets
from tour_recommender.data import FeatureStore


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if hasattr(obj, 'isoformat'): return obj.isoformat()
        return super().default(obj)


def export_market_data(store):
    """Export market profiles for client-side scoring engine."""
    markets = []
    for _, row in store.market_profiles.iterrows():
        if row.get('country') != 'United States':
            continue
        if row['market_avg_capacity'] == 0 and row['market_avg_guarantee'] == 0:
            continue
        markets.append({
            'm': row['market'],
            'c': row['city'],
            's': row['state'],
            'na': int(row['n_artists_played']),
            'ns': int(row['n_total_shows']),
            'af': round(float(row['market_avg_fill_rate']), 3),
            'ac': round(float(row['market_avg_capacity'])),
            'mc': round(float(row['market_median_capacity'])),
            'ag': round(float(row['market_avg_guarantee'])),
            'an': round(float(row['market_avg_artist_net'])),
            'at': round(float(row['market_avg_ticket_price']), 1),
            't': int(row['market_size_tier_num']),
        })
    return markets


def main():
    parser = argparse.ArgumentParser(description='Train tour recommendation model and export for frontend')
    parser.add_argument('--csv-dir', required=True, help='Directory containing settlement CSV files')
    parser.add_argument('--output', default='./export.json', help='Output JSON path')
    parser.add_argument('--n-cities', type=int, default=12, help='Number of cities to recommend per artist')
    args = parser.parse_args()

    # Find all CSVs
    csv_paths = sorted(glob.glob(os.path.join(args.csv_dir, '*.csv')))
    if not csv_paths:
        print(f"No CSV files found in {args.csv_dir}")
        sys.exit(1)

    print(f"Found {len(csv_paths)} settlement CSVs")
    for p in csv_paths:
        print(f"  - {os.path.basename(p)}")

    # Initialize model
    store, model = initialize(csv_paths=csv_paths)

    # Export market data for client-side scoring
    market_data = export_market_data(store)
    print(f"\nExported {len(market_data)} US markets for client-side engine")

    # Generate pre-computed scenarios for all known artists
    scenarios = {}
    artist_names = store.artist_profiles['artist'].unique()
    print(f"\nGenerating recommendations for {len(artist_names)} known artists:")

    for name in artist_names:
        slug = name.lower().replace(' ', '_').replace("'", '')
        print(f"  → {name}...", end=' ')
        try:
            result = recommend_tour_markets(
                artist_name=name,
                n_cities=args.n_cities,
                optimize_for='balanced',
            )
            scenarios[slug] = result
            net = result['financial_summary']['total_predicted_net']
            print(f"${net:,.0f} predicted net across {args.n_cities} markets")
        except Exception as e:
            print(f"ERROR: {e}")

    # Write output
    output = {
        'markets': market_data,
        'scenarios': scenarios,
        'model_metadata': {
            'n_training_shows': len(store.training_matrix),
            'n_training_artists': len(artist_names),
            'n_markets': len(market_data),
            'fill_auc': round(model.metrics.fill_rate_auc or 0, 3),
            'revenue_r2': round(model.metrics.revenue_r2 or 0, 3),
            'revenue_mae': round(model.metrics.revenue_mae or 0, 0),
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, cls=NumpyEncoder, indent=2)

    print(f"\n✅ Exported to {args.output}")
    print(f"   {len(market_data)} markets, {len(scenarios)} artist scenarios")
    print(f"\nTo update the frontend, copy the 'markets' array into TourApp.jsx MARKETS constant")
    print(f"and the 'scenarios' object into the PRECOMPUTED constant.")


if __name__ == '__main__':
    main()
