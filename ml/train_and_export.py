"""
Train the cross-artist ML model and export results for the Next.js frontend.

Usage:
    python train_and_export.py --csv-dir ./data/settlements --output ../app/data/scenarios.json

This script:
  1. Loads all settlement CSVs from the specified directory
  2. Builds the FeatureStore (artist profiles, market profiles, training matrix)
  3. Trains the four-head ML model (fill, revenue, capacity, guarantee)
  4. Runs predictions for all known artists
  5. Exports market data + pre-computed scenarios + seasonality as JSON for the frontend
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
            # New fields
            'nt': round(float(row.get('market_avg_n_tiers', 0)), 1),
            'mt': round(float(row.get('market_avg_max_ticket_price', 0)), 1),
            'vn': round(float(row.get('market_pct_vs_net', 0)), 2),
            'sp': round(float(row.get('market_avg_split_point', 0))),
            'ps': round(float(row.get('market_peak_season_fill', 0)), 3),
            'os': round(float(row.get('market_off_season_fill', 0)), 3),
            'ap': round(float(row.get('market_avg_artist_pct', 0)), 2),
            'p21': round(float(row.get('market_pct_21_plus', 0)), 2),
        })
    return markets


def export_artist_seasonality(store):
    """Export per-artist monthly averages for seasonality charts."""
    seasonality = {}
    all_shows = store.all_shows
    for artist, group in all_shows.groupby('artist'):
        headline = group[group['is_headline'] == 1]
        if len(headline) < 3:
            continue
        monthly = []
        for month in range(1, 13):
            month_shows = headline[headline['month'] == month]
            if len(month_shows) == 0:
                monthly.append({'m': month, 'n': 0, 'f': 0, 'g': 0, 'r': 0})
            else:
                monthly.append({
                    'm': month,
                    'n': int(len(month_shows)),
                    'f': round(float(month_shows['fill_rate'].mean()), 3),
                    'g': round(float(month_shows['guarantee'].mean())),
                    'r': round(float(month_shows['artist_net'].mean())),
                })
        slug = artist.lower().replace(' ', '_').replace("'", '')
        seasonality[slug] = monthly
    return seasonality


def export_aggregate_seasonality(store):
    """Export aggregate monthly averages by growth phase, for unknown-artist fallback."""
    all_shows = store.all_shows
    artist_profiles = store.artist_profiles

    # Use the growth_phase already computed in artist_profiles
    phase_map = {}
    for _, ap in artist_profiles.iterrows():
        phase = ap.get('growth_phase', 'club') or 'club'
        phase_map.setdefault(phase, []).append(ap['artist'])

    agg_seasonality = {}
    for phase, artists in phase_map.items():
        phase_shows = all_shows[
            all_shows['artist'].isin(artists) & (all_shows['is_headline'] == 1)
        ]
        if len(phase_shows) < 3:
            continue
        monthly = []
        for month in range(1, 13):
            ms = phase_shows[phase_shows['month'] == month]
            if len(ms) == 0:
                monthly.append({'m': month, 'n': 0, 'f': 0, 'g': 0, 'r': 0})
            else:
                monthly.append({
                    'm': month,
                    'n': int(len(ms)),
                    'f': round(float(ms['fill_rate'].mean()), 3),
                    'g': round(float(ms['guarantee'].mean())),
                    'r': round(float(ms['artist_net'].mean())),
                })
        agg_seasonality[phase] = monthly

    return agg_seasonality


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

    # Export seasonality
    seasonality = export_artist_seasonality(store)
    print(f"Exported seasonality data for {len(seasonality)} artists")

    # Export aggregate seasonality by growth phase (fallback for unknown artists)
    agg_seasonality = export_aggregate_seasonality(store)
    print(f"Exported aggregate seasonality for {len(agg_seasonality)} growth phases")

    # Generate pre-computed scenarios for all known artists
    scenarios = {}
    artist_names = store.artist_profiles['artist'].unique()
    print(f"\nGenerating recommendations for {len(artist_names)} known artists:")

    for name in artist_names:
        slug = name.lower().replace(' ', '_').replace("'", '')
        print(f"  -> {name}...", end=' ')
        try:
            result = recommend_tour_markets(
                artist_name=name,
                n_cities=args.n_cities,
                optimize_for='balanced',
            )
            scenarios[slug] = result
            income = result['financial_summary']['total_predicted_income']
            net = result['financial_summary']['total_predicted_net']
            merch = result['financial_summary']['total_predicted_merch']
            print(f"income={income:,.0f} (net={net:,.0f} + merch={merch:,.0f})")
        except Exception as e:
            print(f"ERROR: {e}")

    # Write output
    output = {
        'markets': market_data,
        'scenarios': scenarios,
        'seasonality': seasonality,
        'aggregate_seasonality': agg_seasonality,
        'model_metadata': {
            'n_training_shows': len(store.training_matrix),
            'n_training_artists': len(artist_names),
            'n_markets': len(market_data),
            'n_features': len(model.feature_importances),
            'fill_auc': round(model.metrics.fill_rate_auc or 0, 3),
            'revenue_r2': round(model.metrics.revenue_r2 or 0, 3),
            'revenue_mae': round(model.metrics.revenue_mae or 0, 0),
            'capacity_r2': round(model.metrics.capacity_r2 or 0, 3),
            'guarantee_r2': round(model.metrics.guarantee_r2 or 0, 3),
            'guarantee_mae': round(model.metrics.guarantee_mae or 0, 0),
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, cls=NumpyEncoder, indent=2)

    print(f"\nExported to {args.output}")
    print(f"   {len(market_data)} markets, {len(scenarios)} artist scenarios, {len(seasonality)} seasonality profiles, {len(agg_seasonality)} phase aggregates")
    print(f"\nTo update the frontend, copy the 'markets' array into TourApp.jsx MARKETS constant")
    print(f"and the 'scenarios' object into the PRECOMPUTED constant.")


if __name__ == '__main__':
    main()
