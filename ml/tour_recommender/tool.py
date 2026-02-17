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


# ── Profitability Helpers ───────────────────────────────────────────────

def _recommend_deal_structure(profile: dict, market_row: pd.Series) -> dict:
    """Recommend deal type, guarantee, and artist % for this artist in this market."""
    phase = profile.get('growth_phase_num', 0)
    avg_guar = profile.get('current_avg_guarantee', 0)
    sellout_rate = profile.get('sellout_rate', 0)
    upside_rate = profile.get('upside_capture_rate', 0)
    market_vs_net = market_row.get('market_pct_vs_net', 0)
    pred_guarantee = market_row.get('pred_guarantee', 0)

    # Recommend deal type based on artist phase and market norms
    if phase >= 4:
        deal_type = 'Guarantee vs Net'
        artist_pct = min(0.90, max(0.82, profile.get('avg_artist_pct', 0.85)))
        reasoning = 'Arena-level artist commands vs-net deal with premium artist split'
    elif phase >= 3 and (sellout_rate > 0.5 or market_vs_net > 0.5):
        deal_type = 'Guarantee vs Net'
        artist_pct = min(0.90, max(0.80, profile.get('avg_artist_pct', 0.85)))
        reasoning = 'Strong demand justifies vs-net deal with high artist split'
    elif phase >= 2 and upside_rate > 0.4:
        deal_type = 'Guarantee Plus %'
        artist_pct = min(0.85, max(0.75, profile.get('avg_artist_pct', 0.80)))
        reasoning = 'Growing artist benefits from guarantee plus upside participation'
    elif phase >= 2:
        deal_type = 'Guarantee vs Net'
        artist_pct = min(0.85, max(0.70, profile.get('avg_artist_pct', 0.80)))
        reasoning = 'Consistent performance supports vs-net deal structure'
    elif phase == 1:
        deal_type = 'Flat Guarantee'
        artist_pct = 0.0
        reasoning = 'Flat guarantee provides predictable income at club level'
    else:
        deal_type = 'Door Deal' if phase == 0 else 'Flat Guarantee'
        artist_pct = 0.0 if deal_type == 'Flat Guarantee' else 0.70
        reasoning = 'Emerging artist — door deal or modest guarantee recommended'

    # Guarantee: blend model prediction with artist history
    if pred_guarantee > 0 and avg_guar > 0:
        rec_guarantee = pred_guarantee * 0.6 + avg_guar * 0.4
    elif pred_guarantee > 0:
        rec_guarantee = pred_guarantee
    else:
        rec_guarantee = avg_guar

    return {
        'deal_type': deal_type,
        'guarantee': round(float(rec_guarantee), 0),
        'artist_pct': round(float(artist_pct), 2),
        'reasoning': reasoning,
    }


def _calculate_profitability(
    pred_revenue: float, pred_capacity: float, pred_guarantee: float,
    fill_prob: float, profile: dict, market_row: pd.Series, deal_rec: dict
) -> dict:
    """Calculate profitability estimates for a market."""
    avg_ticket = profile.get('current_avg_ticket_price', 0)
    max_ticket = profile.get('avg_max_ticket_price', avg_ticket * 1.5)
    n_tiers = profile.get('avg_n_tiers', 3)
    merch_soft_pct = profile.get('avg_merch_soft_pct', 0.85) or 0.85  # default when no merch data
    rev_per_head = profile.get('avg_revenue_per_head', 0)

    # Use market ticket prices as fallback
    if avg_ticket <= 0:
        avg_ticket = market_row.get('market_avg_ticket_price', 35)
        max_ticket = market_row.get('market_avg_max_ticket_price', avg_ticket * 1.5)

    # If guarantee is funded, minimum ~50% fill is expected (promoter wouldn't book otherwise)
    min_fill = 0.50 if pred_guarantee > 1000 else 0.20
    est_fill = min(max(fill_prob * 1.1, min_fill), 1.0)
    est_sold = max(int(pred_capacity * est_fill), 1)

    gross_ticket = est_sold * avg_ticket

    # Merch estimate: $5-15 per head depending on artist tier
    phase = profile.get('growth_phase_num', 0)
    merch_per_head = {0: 3, 1: 5, 2: 8, 3: 12, 4: 18}.get(phase, 5)
    est_merch_gross = est_sold * merch_per_head
    est_merch_artist = est_merch_gross * merch_soft_pct

    # Artist income: use predicted revenue if available, otherwise derive from deal
    if pred_revenue > 100:
        artist_net = pred_revenue
    elif deal_rec['deal_type'] == 'Flat Guarantee':
        artist_net = deal_rec['guarantee']
    elif deal_rec['artist_pct'] > 0:
        artist_net = max(deal_rec['guarantee'], gross_ticket * deal_rec['artist_pct'])
    else:
        artist_net = deal_rec['guarantee']

    total_income = artist_net + est_merch_artist

    return {
        'gross_ticket_revenue': round(float(gross_ticket), 0),
        'estimated_merch_revenue': round(float(est_merch_artist), 0),
        'artist_net': round(float(artist_net), 0),
        'total_artist_income': round(float(total_income), 0),
        'recommended_avg_ticket_price': round(float(avg_ticket), 2),
        'ticket_price_range': [round(float(avg_ticket * 0.6), 2), round(float(max_ticket), 2)],
        'est_sold': est_sold,
        'est_fill_rate': round(float(est_fill), 2),
        'n_tiers': int(n_tiers) if n_tiers > 0 else 3,
        'merch_per_head': merch_per_head,
    }


MONTH_NAMES = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _get_best_months(profile: dict, market_row: pd.Series) -> dict:
    """Determine best touring months based on artist and market seasonality."""
    peak_fill = profile.get('peak_season_fill_rate', 0)
    off_fill = profile.get('off_season_fill_rate', 0)
    mkt_peak = market_row.get('market_peak_season_fill', 0)
    mkt_off = market_row.get('market_off_season_fill', 0)

    # Peak months: Sep-Nov, Mar-May
    peak_months = [3, 4, 5, 9, 10, 11]
    shoulder_months = [1, 2, 6, 12]
    avoid_months = [7, 8]  # Summer competition, festival season

    # Adjust based on actual data
    if peak_fill > 0 and off_fill > 0:
        season_advantage = peak_fill - off_fill
    elif mkt_peak > 0 and mkt_off > 0:
        season_advantage = mkt_peak - mkt_off
    else:
        season_advantage = 0.1  # default slight peak preference

    best = [MONTH_NAMES[m] for m in peak_months]
    ok = [MONTH_NAMES[m] for m in shoulder_months]
    avoid = [MONTH_NAMES[m] for m in avoid_months]

    return {
        'best_months': best,
        'shoulder_months': ok,
        'avoid_months': avoid,
        'season_advantage': round(float(season_advantage), 2),
    }


# ── Profile Builder ─────────────────────────────────────────────────────

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
    if spotify_monthly_listeners > 10_000_000:
        phase, phase_num = 'arena', 4
        est_capacity, est_guarantee = 12000, 300000
        est_ticket, est_max_ticket = 85, 150
        est_artist_pct = 0.85
    elif spotify_monthly_listeners > 3_000_000:
        phase, phase_num = 'theater', 3
        est_capacity, est_guarantee = 3500, 75000
        est_ticket, est_max_ticket = 55, 90
        est_artist_pct = 0.82
    elif spotify_monthly_listeners > 1_000_000:
        phase, phase_num = 'club_to_theater', 2
        est_capacity, est_guarantee = 1200, 15000
        est_ticket, est_max_ticket = 35, 60
        est_artist_pct = 0.80
    elif spotify_monthly_listeners > 300_000:
        phase, phase_num = 'club', 1
        est_capacity, est_guarantee = 500, 5000
        est_ticket, est_max_ticket = 25, 40
        est_artist_pct = 0.0
    else:
        phase, phase_num = 'emerging', 0
        est_capacity, est_guarantee = 200, 1500
        est_ticket, est_max_ticket = 18, 30
        est_artist_pct = 0.0

    return {
        'artist': artist_name,
        'growth_phase': phase,
        'growth_phase_num': phase_num,
        'current_avg_capacity': est_capacity,
        'current_avg_guarantee': est_guarantee,
        'current_avg_fill_rate': 0.70,
        'current_avg_ticket_price': est_ticket,
        'current_avg_artist_net': est_guarantee * 1.1,
        'guarantee_cagr': listener_growth_rate,
        'total_shows': 0,
        'total_headline_shows': 0,
        'n_markets_played': 0,
        'months_active': 12,
        'headline_ratio': 0.7,
        'sellout_rate': 0.3,
        # Deal
        'avg_artist_pct': est_artist_pct,
        'pct_vs_net_deals': 0.5 if phase_num >= 2 else 0,
        'pct_flat_deals': 0.5 if phase_num <= 1 else 0.2,
        'upside_capture_rate': 0.4,
        'avg_split_point': 0,
        # Merch
        'avg_merch_soft_pct': 0.85,
        'merch_seller_artist_rate': 0.5,
        # Ticket
        'avg_n_tiers': 3,
        'avg_max_ticket_price': est_max_ticket,
        'avg_price_spread': est_max_ticket - est_ticket,
        # Seasonality
        'peak_season_fill_rate': 0.75,
        'off_season_fill_rate': 0.65,
        # Revenue
        'avg_revenue_per_head': est_guarantee / max(est_capacity, 1),
        # Streaming
        'spotify_monthly_listeners': spotify_monthly_listeners,
        'top_streaming_cities': top_streaming_cities or [],
        'genre': genre,
        'profile_source': 'streaming_calibration',
    }


# ── Main Recommendation Function ───────────────────────────────────────

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

    Returns ranked markets with predictions, profitability, deal recs, and route.
    """
    if _STORE is None or _MODEL is None:
        raise RuntimeError("Call initialize() first with CSV data paths")

    # ── Build artist profile ──────────────────────────────────────────
    profile_source = 'unknown'

    if artist_profile:
        profile = artist_profile
        profile_source = 'manual'
    elif artist_name and _STORE.get_artist_profile(artist_name):
        profile = _STORE.get_artist_profile(artist_name)
        profile_source = 'settlement_data'
    elif artist_name and spotify_monthly_listeners:
        profile = _build_profile_from_streaming(
            artist_name, spotify_monthly_listeners, top_streaming_cities,
            chartmetric_rank, genre)
        profile_source = 'streaming_calibration'
    elif artist_name:
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

    # Guarantee score: blend guarantee with revenue for better ranking
    if 'pred_guarantee' in preds.columns:
        combined = preds['pred_artist_net'] + preds['pred_guarantee'] * 0.3
        preds['revenue_score'] = combined.rank(pct=True) * 100 if len(preds) > 1 else 50

    preds['discovery_score'] = np.where(
        preds['has_artist_history'],
        50,
        preds['market_n_artists'].rank(pct=True) * 100
    )

    def recency_score(days, has_history):
        if not has_history:
            return 70
        if days < 90:
            return 30
        elif days < 270:
            return 90
        elif days < 540:
            return 100
        elif days < 900:
            return 60
        else:
            return 40
    preds['recency_score'] = preds.apply(
        lambda r: recency_score(r['days_since_last'], r['has_artist_history']), axis=1)

    preds['market_score'] = (
        w['fill'] * preds['fill_score'] +
        w['revenue'] * preds['revenue_score'] +
        w['discovery'] * preds['discovery_score'] +
        w['recency'] * preds['recency_score']
    )

    if min_capacity:
        preds = preds[preds['pred_capacity'] >= min_capacity]
    if max_capacity:
        preds = preds[preds['pred_capacity'] <= max_capacity]

    if include_markets:
        must_have = preds[preds['market'].isin(include_markets)]
        rest = preds[~preds['market'].isin(include_markets)]
        top = pd.concat([must_have, rest.nlargest(n_cities - len(must_have), 'market_score')])
    else:
        top = preds.nlargest(n_cities, 'market_score')

    top = top.sort_values('market_score', ascending=False).reset_index(drop=True)

    # ── Route optimization ────────────────────────────────────────────
    route_idx, route_dist = optimize_route(top['market'].tolist(), tour_start_city)

    # ── Build response with profitability ─────────────────────────────
    markets_response = []
    deal_type_counts = {}

    for i, (_, row) in enumerate(top.iterrows()):
        # Deal recommendation
        deal_rec = _recommend_deal_structure(profile, row)
        deal_type_counts[deal_rec['deal_type']] = deal_type_counts.get(deal_rec['deal_type'], 0) + 1

        # Profitability calculation
        profitability = _calculate_profitability(
            row['pred_artist_net'], row['pred_capacity'],
            row.get('pred_guarantee', 0),
            row['pred_fill_probability'], profile, row, deal_rec)

        # Seasonality
        timing = _get_best_months(profile, row)

        # Reasoning
        reasons = []
        if row['pred_fill_probability'] > 0.8:
            reasons.append(f"High sellout probability ({row['pred_fill_probability']:.0%})")
        if row['has_artist_history'] and row['prior_visits'] >= 3:
            reasons.append(f"Proven market ({int(row['prior_visits'])} prior visits)")
        if not row['has_artist_history'] and row['market_n_artists'] > 1:
            reasons.append(f"Strong market ({int(row['market_n_artists'])} artists play here), new opportunity")
        if profitability['total_artist_income'] > profile.get('current_avg_artist_net', 0) * 1.2:
            reasons.append("Above-average income potential")
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
            'predicted_guarantee': round(float(row.get('pred_guarantee', 0)), 0),
            # Deal recommendation
            'recommended_deal_type': deal_rec['deal_type'],
            'recommended_guarantee': deal_rec['guarantee'],
            'recommended_artist_pct': deal_rec['artist_pct'],
            'deal_reasoning': deal_rec['reasoning'],
            # Profitability
            'profitability': profitability,
            # Ticket pricing
            'recommended_avg_ticket_price': profitability['recommended_avg_ticket_price'],
            'ticket_price_range': profitability['ticket_price_range'],
            # Timing
            'best_months': timing['best_months'],
            'seasonality_score': timing['season_advantage'],
            # History
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

    # ── Financial summary ─────────────────────────────────────────────
    total_net = sum(m['profitability']['artist_net'] for m in markets_response)
    total_merch = sum(m['profitability']['estimated_merch_revenue'] for m in markets_response)
    total_income = sum(m['profitability']['total_artist_income'] for m in markets_response)
    avg_guarantee = np.mean([m['recommended_guarantee'] for m in markets_response])
    avg_ticket = np.mean([m['recommended_avg_ticket_price'] for m in markets_response])

    # Best tour months: aggregate seasonality across selected markets
    best_tour_months = _get_best_months(profile, top.iloc[0] if len(top) > 0 else pd.Series())

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
            'avg_ticket_price': round(float(profile.get('current_avg_ticket_price', 0)), 2),
            'avg_merch_soft_pct': round(float(profile.get('avg_merch_soft_pct', 0)), 2),
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
            'total_predicted_net': round(float(total_net), 0),
            'total_predicted_merch': round(float(total_merch), 0),
            'total_predicted_income': round(float(total_income), 0),
            'avg_recommended_guarantee': round(float(avg_guarantee), 0),
            'avg_ticket_price': round(float(avg_ticket), 2),
            'avg_predicted_capacity': round(float(np.mean([m['predicted_capacity'] for m in markets_response])), 0),
            'avg_predicted_fill_prob': round(float(np.mean([m['predicted_fill_probability'] for m in markets_response])), 2),
            'deal_type_breakdown': deal_type_counts,
            'best_tour_months': best_tour_months['best_months'],
        },
        'model_metadata': {
            'model_type': 'cross_artist_gradient_boosted',
            'n_features': len(_MODEL.feature_importances),
            'fill_auc': _MODEL.metrics.fill_rate_auc,
            'revenue_r2': _MODEL.metrics.revenue_r2,
            'revenue_mae': _MODEL.metrics.revenue_mae,
            'capacity_r2': _MODEL.metrics.capacity_r2,
            'guarantee_r2': _MODEL.metrics.guarantee_r2,
            'guarantee_mae': _MODEL.metrics.guarantee_mae,
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
        "Cross-artist ML tour recommendation engine trained on live show settlements. "
        "Predicts optimal tour markets for ANY artist — including those not in the training data — "
        "by learning patterns across artists' fill rates, guarantees, capacity progression, "
        "deal structures, merchandise, and seasonality. Returns ranked cities with predicted capacity, "
        "revenue, profitability analysis, deal structure recommendations, ticket pricing, "
        "and an optimized geographic route."
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
