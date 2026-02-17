#!/usr/bin/env python3
"""
demo.py - Cross-Artist Tour Recommendation Engine

Demonstrates three modes:
1. KNOWN ARTIST (Laufey) - uses settlement data
2. KNOWN ARTIST (Peter McPoland) - uses settlement data
3. UNKNOWN ARTIST - uses Spotify/Chartmetric streaming data only
"""

import sys, json
sys.path.insert(0, '/home/claude/tour-recommender')

from tour_recommender import initialize, recommend_tour_markets

# ── Initialize: load ALL CSVs and train the cross-artist model ──────────

print("=" * 90)
print("INITIALIZING: Loading settlement data and training cross-artist model...")
print("=" * 90)

store, model = initialize(csv_paths=[
    '/mnt/user-data/uploads/laufey.csv',
    '/mnt/user-data/uploads/petermcpoland.csv',
])

print(f"\nFeature Store:")
print(f"  Artists: {len(store.artist_profiles)}")
print(f"  Markets: {len(store.market_profiles)}")
print(f"  Training rows: {len(store.training_matrix):,}")
print(f"  Artist×Market interactions: {len(store.artist_market_history)}")


def print_result(result):
    """Pretty-print a recommendation result."""
    ap = result['artist_profile']
    print(f"\n🎵 {result['artist']} ({result['profile_source']})")
    print(f"   Phase: {ap['growth_phase']} | Avg Cap: {ap['current_avg_capacity']:,.0f} | "
          f"Avg Guarantee: ${ap['current_avg_guarantee']:,.0f} | Fill: {ap['avg_fill_rate']:.0%}")

    print(f"\n{'Rk':<4} {'Market':<25} {'Score':<7} {'Fill%':<7} {'Pred Cap':<10} {'Pred Net':<14} {'History':<30}")
    print("─" * 97)
    for m in result['markets']:
        hist = f"({m['historical']['prior_visits']} visits, last {m['historical']['days_since_last_visit']}d ago)" if m['has_artist_history'] and m['historical'] else "(NEW MARKET)"
        print(f"{m['rank']:<4} {m['market']:<25} {m['market_score']:<7.1f} {m['predicted_fill_probability']:<7.0%} "
              f"{m['predicted_capacity']:<10,} ${m['predicted_net_revenue']:<13,.0f} {hist}")

    print(f"\n🗺️  Route ({result['route']['total_distance_miles']:,} miles): "
          + " → ".join(result['route']['routed_markets']))

    fs = result['financial_summary']
    print(f"\n💰 Total Predicted Net: ${fs['total_predicted_net']:,.0f} | "
          f"Avg Capacity: {fs['avg_predicted_capacity']:,.0f} | "
          f"Avg Fill Prob: {fs['avg_predicted_fill_prob']:.0%}")

    mm = result['model_metadata']
    print(f"\n🧠 Model: {mm['n_training_shows']} shows, {mm['n_training_artists']} artists | "
          f"Revenue R²={mm['revenue_r2']} | Fill AUC={mm['fill_auc']} | Capacity R²={mm['capacity_r2']}")


# ══════════════════════════════════════════════════════════════════════════
# MODE 1: KNOWN ARTIST — Laufey (arena-level, in training data)
# ══════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 90)
print("MODE 1: KNOWN ARTIST — Laufey (arena-level, has settlement data)")
print("═" * 90)

laufey = recommend_tour_markets(
    artist_name="Laufey",
    n_cities=10,
    optimize_for="balanced",
    tour_start_city="New York, NY",
)
print_result(laufey)


# ══════════════════════════════════════════════════════════════════════════
# MODE 2: KNOWN ARTIST — Peter McPoland (club-to-theater, growth focus)
# ══════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 90)
print("MODE 2: KNOWN ARTIST — Peter McPoland (club-to-theater, optimize for growth)")
print("═" * 90)

peter = recommend_tour_markets(
    artist_name="Peter McPoland",
    n_cities=10,
    optimize_for="growth",
    tour_start_city="Dallas, TX",
)
print_result(peter)


# ══════════════════════════════════════════════════════════════════════════
# MODE 3: UNKNOWN ARTIST — Someone NOT in the data at all
# The model uses cross-artist patterns + streaming calibration
# ══════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 90)
print("MODE 3: UNKNOWN ARTIST — 'Indie Artist X' (NOT in training data)")
print("  Using only: spotify_monthly_listeners=2,000,000 + genre='indie-folk'")
print("  The model recommends based on patterns from ALL artists in training data")
print("═" * 90)

unknown = recommend_tour_markets(
    artist_name="Indie Artist X",
    spotify_monthly_listeners=2_000_000,
    top_streaming_cities=["Austin", "Nashville", "Portland", "Brooklyn"],
    genre="indie-folk",
    n_cities=10,
    optimize_for="balanced",
    tour_start_city="Nashville, TN",
)
print_result(unknown)


# ══════════════════════════════════════════════════════════════════════════
# COMPARISON: Same market, different artists
# ══════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 90)
print("CROSS-ARTIST COMPARISON: How does the model differentiate?")
print("═" * 90)

# Find common markets
laufey_markets = {m['market']: m for m in laufey['markets']}
peter_markets = {m['market']: m for m in peter['markets']}
unknown_markets = {m['market']: m for m in unknown['markets']}

all_in_common = set(laufey_markets) & set(peter_markets) & set(unknown_markets)
print(f"\nMarkets recommended for ALL three artists: {all_in_common or 'None'}")
print(f"Markets unique to Laufey: {set(laufey_markets) - set(peter_markets) - set(unknown_markets)}")
print(f"Markets unique to Peter: {set(peter_markets) - set(laufey_markets) - set(unknown_markets)}")
print(f"Markets unique to Unknown: {set(unknown_markets) - set(laufey_markets) - set(peter_markets)}")


# ══════════════════════════════════════════════════════════════════════════
# SAMPLE JSON OUTPUT (what the synthesizer receives)
# ══════════════════════════════════════════════════════════════════════════

print("\n\n" + "═" * 90)
print("SAMPLE TOOL OUTPUT (Unknown Artist — first 2 markets)")
print("═" * 90)
sample = {k: v for k, v in unknown.items() if k != 'markets'}
sample['markets'] = unknown['markets'][:2]
print(json.dumps(sample, indent=2, default=str))
