"""
tour_recommender/tool.py
The tool interface for the LangGraph orchestrator.
This is what gets called: recommend_tour_markets(artist_profile, ...)

Supports three modes:
1. KNOWN artist (has settlement data) → uses real historical features
2. UNKNOWN artist with streaming data → maps Spotify/Chartmetric to profile
3. UNKNOWN artist with manual profile → uses provided features directly
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from .data import FeatureStore
from .model import TourRecommendationModel, optimize_route


# ── Singleton store + model (initialized once, reused) ──────────────────

_STORE: Optional[FeatureStore] = None
_MODEL: Optional[TourRecommendationModel] = None


def initialize(csv_paths: list[str] = None, csv_dir: str = None):
    """Initialize the feature store and train the model. Call once at startup."""
    global _STORE, _MODEL
    _STORE = FeatureStore.build(csv_paths=csv_paths, csv_dir=csv_dir)
    _MODEL = TourRecommendationModel()
    _MODEL.train(_STORE)
    return _STORE, _MODEL


def _build_profile_from_streaming(
    artist_name: str,
    spotify_monthly_listeners: int = 0,
    top_streaming_cities: list[str] = None,
    chartmetric_rank: int = None,
    genre: str = None,
    listener_growth_rate: float = 0,
) -> dict:
    """
    Build an artist profile from streaming data when no settlement data exists.
    Uses calibration heuristics (will be replaced by calibration model at scale).
    """
    # Rough calibration: Spotify listeners → capacity/guarantee tier
    # These thresholds will be learned from artists where we have both data sources
    if spotify_monthly_listeners > 10_000_000:
        phase, phase_num = 'arena', 4
        est_capacity = 12000
        est_guarantee = 300000
    elif spotify_monthly_listeners > 3_000_000:
        phase, phase_num = 'theater', 3
        est_capacity = 3500
        est_guarantee = 75000
    elif spotify_monthly_listeners > 1_000_000:
        phase, phase_num = 'club_to_theater', 2
        est_capacity = 1200
        est_guarantee = 15000
    elif spotify_monthly_listeners > 300_000:
        phase, phase_num = 'club', 1
        est_capacity = 500
        est_guarantee = 5000
    else:
        phase, phase_num = 'emerging', 0
        est_capacity = 200
        est_guarantee = 1500

    return {
        'artist': artist_name,
        'growth_phase': phase,
        'growth_phase_num': phase_num,
        'current_avg_capacity': est_capacity,
        'current_avg_guarantee': est_guarantee,
        'current_avg_fill_rate': 0.70,  # conservative default
        'current_avg_ticket_price': est_guarantee / max(est_capacity, 1) * 3,
        'current_avg_artist_net': est_guarantee * 1.1,
        'guarantee_cagr': listener_growth_rate,
        'total_shows': 0,
        'total_headline_shows': 0,
        'n_markets_played': 0,
        'months_active': 12,
        'headline_ratio': 0.7,
        'sellout_rate': 0.3,
        'spotify_monthly_listeners': spotify_monthly_listeners,
        'top_streaming_cities': top_streaming_cities or [],
        'genre': genre,
        'profile_source': 'streaming_calibration',
    }


def recommend_tour_markets(
    # Artist identification (one of these must be provided)
    artist_name: str = None,
    artist_profile: dict = None,  # Manual profile override

    # Streaming data for unknown artists
    spotify_monthly_listeners: int = None,
    top_streaming_cities: list[str] = None,
    chartmetric_rank: int = None,
    genre: str = None,

    # Tour parameters
    n_cities: int = 10,
    region: str = 'us',
    optimize_for: str = 'balanced',
    min_capacity: int = None,
    max_capacity: int = None,
    exclude_markets: list[str] = None,
    include_markets: list[str] = None,
    tour_start_city: str = None,
) -> dict:
    """
    ML-powered tour market recommendation. Works for ANY artist.

    Three modes:
    1. artist_name matches settlement data → full historical features
    2. artist_name + spotify data → streaming-calibrated profile
    3. artist_profile dict → use provided features directly

    Returns ranked markets with predictions, route, and financials.
    """
    if _STORE is None or _MODEL is None:
        raise RuntimeError("Call initialize() first with CSV data paths")

    # ── Build artist profile ──────────────────────────────────────────
    profile_source = 'unknown'

    if artist_profile:
        # Mode 3: Manual profile provided
        profile = artist_profile
        profile_source = 'manual'
    elif artist_name and _STORE.get_artist_profile(artist_name):
        # Mode 1: Known artist with settlement data
        profile = _STORE.get_artist_profile(artist_name)
        profile_source = 'settlement_data'
    elif artist_name and spotify_monthly_listeners:
        # Mode 2: Unknown artist with streaming data
        profile = _build_profile_from_streaming(
            artist_name, spotify_monthly_listeners, top_streaming_cities,
            chartmetric_rank, genre)
        profile_source = 'streaming_calibration'
    elif artist_name:
        # Mode 2b: Unknown artist, no streaming data → minimum viable profile
        profile = _build_profile_from_streaming(artist_name, 500_000)
        profile_source = 'default_estimate'
    else:
        raise ValueError("Provide artist_name or artist_profile")

    # ── Get candidate markets ─────────────────────────────────────────
    all_markets = _STORE.market_profiles.copy()
    if region == 'us':
        all_markets = all_markets[all_markets['country'] == 'United States']
    elif region == 'north_america':
        all_markets = all_markets[all_markets['country'].isin(['United States', 'Canada'])]
    if exclude_markets:
        all_markets = all_markets[~all_markets['market'].isin(exclude_markets)]

    market_list = all_markets['market'].tolist()

    # ── Run ML predictions ────────────────────────────────────────────
    predictions = _MODEL.predict_for_markets(profile, _STORE, market_list)

    # ── Score and rank ────────────────────────────────────────────────
    weights = {
        'balanced': {'fill': 0.30, 'revenue': 0.35, 'discovery': 0.15, 'recency': 0.20},
        'revenue':  {'fill': 0.20, 'revenue': 0.50, 'discovery': 0.10, 'recency': 0.20},
        'growth':   {'fill': 0.25, 'revenue': 0.15, 'discovery': 0.40, 'recency': 0.20},
    }
    w = weights.get(optimize_for, weights['balanced'])

    preds = predictions.copy()

    # Normalize each signal to 0-100
    preds['fill_score'] = preds['pred_fill_probability'] * 100
    preds['revenue_score'] = preds['pred_artist_net'].rank(pct=True) * 100 if len(preds) > 1 else 50

    # Discovery score: bonus for markets with many artists (proven) but not yet visited
    preds['discovery_score'] = np.where(
        preds['has_artist_history'],
        50,  # Known market — neutral
        preds['market_n_artists'].rank(pct=True) * 100  # Unknown market — rank by how many artists play there
    )

    # Recency score: if artist has played here, prefer optimal gap (6-18 months)
    def recency_score(days, has_history):
        if not has_history:
            return 70  # neutral for new markets
        if days < 90:
            return 30   # too recent
        elif days < 270:
            return 90   # great timing
        elif days < 540:
            return 100   # ideal return window
        elif days < 900:
            return 60   # getting stale
        else:
            return 40   # very stale
    preds['recency_score'] = preds.apply(
        lambda r: recency_score(r['days_since_last'], r['has_artist_history']), axis=1)

    preds['market_score'] = (
        w['fill'] * preds['fill_score'] +
        w['revenue'] * preds['revenue_score'] +
        w['discovery'] * preds['discovery_score'] +
        w['recency'] * preds['recency_score']
    )

    # Apply capacity filters
    if min_capacity:
        preds = preds[preds['pred_capacity'] >= min_capacity]
    if max_capacity:
        preds = preds[preds['pred_capacity'] <= max_capacity]

    # Handle must-include markets
    if include_markets:
        must_have = preds[preds['market'].isin(include_markets)]
        rest = preds[~preds['market'].isin(include_markets)]
        top = pd.concat([must_have, rest.nlargest(n_cities - len(must_have), 'market_score')])
    else:
        top = preds.nlargest(n_cities, 'market_score')

    top = top.sort_values('market_score', ascending=False).reset_index(drop=True)

    # ── Route optimization ────────────────────────────────────────────
    route_idx, route_dist = optimize_route(top['market'].tolist(), tour_start_city)

    # ── Build response ────────────────────────────────────────────────
    markets_response = []
    for i, (_, row) in enumerate(top.iterrows()):
        # Reasoning
        reasons = []
        if row['pred_fill_probability'] > 0.8:
            reasons.append(f"High sellout probability ({row['pred_fill_probability']:.0%})")
        if row['has_artist_history'] and row['prior_visits'] >= 3:
            reasons.append(f"Proven market ({int(row['prior_visits'])} prior visits)")
        if not row['has_artist_history'] and row['market_n_artists'] > 1:
            reasons.append(f"Strong market ({int(row['market_n_artists'])} artists play here), new opportunity")
        if row['pred_artist_net'] > profile.get('current_avg_artist_net', 0) * 1.2:
            reasons.append("Above-average revenue potential")
        if row['has_artist_history'] and 270 < row['days_since_last'] < 540:
            reasons.append("Ideal return timing (building anticipation)")
        if not reasons:
            reasons.append("Recommended by cross-artist demand model")

        markets_response.append({
            'rank': i + 1,
            'city': row['city'],
            'state': row['state'],
            'market': row['market'],
            'country': row['country'],
            'market_score': round(float(row['market_score']), 1),
            'confidence': round(float(row['pred_fill_probability']), 2),
            'predicted_capacity': int(round(row['pred_capacity'])),
            'predicted_net_revenue': round(float(row['pred_artist_net']), 0),
            'predicted_fill_probability': round(float(row['pred_fill_probability']), 2),
            'has_artist_history': bool(row['has_artist_history']),
            'historical': {
                'prior_visits': int(row['prior_visits']),
                'prior_avg_fill_rate': round(float(row['prior_avg_fill']), 2),
                'prior_last_capacity': int(row['prior_last_capacity']),
                'prior_last_guarantee': round(float(row['prior_last_guarantee']), 0),
                'days_since_last_visit': int(row['days_since_last']) if row['has_artist_history'] else None,
            } if row['has_artist_history'] else None,
            'reasoning': '. '.join(reasons) + '.',
        })

    routed_markets = [markets_response[i] for i in route_idx]

    return {
        'artist': artist_name or profile.get('artist', 'Unknown'),
        'profile_source': profile_source,
        'artist_profile': {
            'growth_phase': profile.get('growth_phase', 'unknown'),
            'current_avg_capacity': round(float(profile.get('current_avg_capacity', 0)), 0),
            'current_avg_guarantee': round(float(profile.get('current_avg_guarantee', 0)), 0),
            'avg_fill_rate': round(float(profile.get('current_avg_fill_rate', 0)), 2),
            'guarantee_cagr': round(float(profile.get('guarantee_cagr', 0)), 2),
            'total_headline_shows': int(profile.get('total_headline_shows', 0)),
            'n_markets_played': int(profile.get('n_markets_played', 0)),
        },
        'recommendation_config': {
            'optimize_for': optimize_for,
            'region': region,
            'n_requested': n_cities,
        },
        'markets': markets_response,
        'route': {
            'order': route_idx,
            'routed_markets': [m['market'] for m in routed_markets],
            'total_distance_miles': route_dist,
            'start_city': tour_start_city or (routed_markets[0]['market'] if routed_markets else None),
        },
        'financial_summary': {
            'total_predicted_net': round(sum(m['predicted_net_revenue'] for m in markets_response), 0),
            'avg_predicted_capacity': round(np.mean([m['predicted_capacity'] for m in markets_response]), 0),
            'avg_predicted_fill_prob': round(np.mean([m['predicted_fill_probability'] for m in markets_response]), 2),
        },
        'model_metadata': {
            'model_type': 'cross_artist_gradient_boosted',
            'fill_auc': _MODEL.metrics.fill_rate_auc,
            'revenue_r2': _MODEL.metrics.revenue_r2,
            'revenue_mae': _MODEL.metrics.revenue_mae,
            'capacity_r2': _MODEL.metrics.capacity_r2,
            'n_training_shows': _MODEL.metrics.n_train,
            'n_training_artists': _MODEL.metrics.n_artists,
            'top_features': dict(sorted(
                _MODEL.feature_importances.items(), key=lambda x: -x[1])[:10]),
            'data_freshness': datetime.now().strftime('%Y-%m-%d'),
            'markets_in_model': len(_STORE.market_profiles),
        }
    }


# ── LangGraph Tool Definition ──────────────────────────────────────────

TOOL_DEFINITION = {
    "name": "recommend_tour_markets",
    "description": (
        "Cross-artist ML tour recommendation engine trained on 100K+ live show settlements. "
        "Predicts optimal tour markets for ANY artist — including those not in the training data — "
        "by learning patterns across hundreds of artists' fill rates, guarantees, capacity progression, "
        "and market demand. Accepts artist profiles from settlement data, Spotify/Chartmetric, "
        "or manual specification. Returns ranked cities with predicted capacity, revenue, fill "
        "probability, and an optimized geographic route."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "artist_name": {
                "type": "string",
                "description": "Artist name. If in training data, uses real history. Otherwise builds profile from streaming data."
            },
            "spotify_monthly_listeners": {
                "type": "integer",
                "description": "Spotify monthly listeners (for unknown artists). Used to estimate tier."
            },
            "top_streaming_cities": {
                "type": "array", "items": {"type": "string"},
                "description": "Top cities from Spotify/Chartmetric for market affinity."
            },
            "genre": {
                "type": "string",
                "description": "Artist genre (for cold-start recommendations)."
            },
            "n_cities": {"type": "integer", "default": 10},
            "region": {"type": "string", "enum": ["us", "north_america", "global"], "default": "us"},
            "optimize_for": {"type": "string", "enum": ["revenue", "growth", "balanced"], "default": "balanced"},
            "min_capacity": {"type": "integer"},
            "max_capacity": {"type": "integer"},
            "exclude_markets": {"type": "array", "items": {"type": "string"}},
            "include_markets": {"type": "array", "items": {"type": "string"}},
            "tour_start_city": {"type": "string"},
        },
        "required": ["artist_name"]
    }
}
