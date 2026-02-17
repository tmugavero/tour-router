"""
tour_recommender/model.py
Cross-artist ML model: trained on ALL settlement data, predicts for ANY artist.
"""

import numpy as np
import pandas as pd
import math
import joblib
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error
from typing import Optional
from dataclasses import dataclass

from .data import FeatureStore


# ── Feature columns used by models ─────────────────────────────────────

ARTIST_FEATURES = [
    'a_growth_phase', 'a_avg_capacity', 'a_avg_guarantee', 'a_avg_fill_rate',
    'a_avg_ticket_price', 'a_avg_artist_net', 'a_total_shows',
    'a_headline_shows', 'a_n_markets', 'a_months_active', 'a_sellout_rate',
    'a_avg_revenue_per_head',
]

ARTIST_MARKET_FEATURES = [
    'am_prior_visits', 'am_prior_headline_visits', 'am_prior_avg_fill',
    'am_prior_avg_capacity', 'am_prior_last_capacity', 'am_prior_avg_guarantee',
    'am_days_since_last', 'am_prior_sellout_rate', 'am_has_history',
]

MARKET_FEATURES = [
    'market_avg_fill_rate', 'market_avg_capacity', 'market_median_capacity',
    'market_avg_guarantee', 'market_median_guarantee', 'market_avg_ticket_price',
    'market_avg_artist_net', 'market_size_tier_num', 'n_artists_played',
    'n_total_shows',
]

DEAL_FEATURES = [
    'deal_is_flat', 'deal_is_vs_net', 'deal_artist_pct',
    'deal_split_point_ratio', 'deal_has_upside',
]

SHOW_CONTEXT_FEATURES = [
    'show_month', 'show_quarter', 'show_is_weekend', 'show_is_peak_season',
    'show_n_tiers', 'show_avg_ticket_price', 'show_max_ticket_price',
    'show_is_21_plus',
]

ALL_FEATURES = (ARTIST_FEATURES + ARTIST_MARKET_FEATURES + MARKET_FEATURES
                + DEAL_FEATURES + SHOW_CONTEXT_FEATURES)


# ── Coordinates for routing ─────────────────────────────────────────────

MARKET_COORDS = {
    'New York, NY': (40.7128, -74.0060), 'Los Angeles, CA': (34.0522, -118.2437),
    'Chicago, IL': (41.8781, -87.6298), 'Houston, TX': (29.7604, -95.3698),
    'Phoenix, AZ': (33.4484, -112.0740), 'Philadelphia, PA': (39.9526, -75.1652),
    'San Diego, CA': (32.7157, -117.1611), 'Dallas, TX': (32.7767, -96.7970),
    'Austin, TX': (30.2672, -97.7431), 'San Francisco, CA': (37.7749, -122.4194),
    'Seattle, WA': (47.6062, -122.3321), 'Denver, CO': (39.7392, -104.9903),
    'Washington, DC': (38.9072, -77.0369), 'Nashville, TN': (36.1627, -86.7816),
    'Boston, MA': (42.3601, -71.0589), 'Atlanta, GA': (33.7490, -84.3880),
    'Portland, OR': (45.5152, -122.6784), 'Minneapolis, MN': (44.9778, -93.2650),
    'Salt Lake City, UT': (40.7608, -111.8910), 'Detroit, MI': (42.3314, -83.0458),
    'Charlotte, NC': (35.2271, -80.8431), 'Raleigh, NC': (35.7796, -78.6382),
    'Miami, FL': (25.7617, -80.1918), 'Tampa, FL': (27.9506, -82.4572),
    'Orlando, FL': (28.5383, -81.3792), 'Pittsburgh, PA': (40.4406, -79.9959),
    'Cleveland, OH': (41.4993, -81.6944), 'Columbus, OH': (39.9612, -82.9988),
    'Indianapolis, IN': (39.7684, -86.1581), 'Kansas City, MO': (39.0997, -94.5786),
    'St. Louis, MO': (38.6270, -90.1994), 'New Orleans, LA': (29.9511, -90.0715),
    'Richmond, VA': (37.5407, -77.4360), 'Santa Ana, CA': (33.7455, -117.8677),
    'West Hollywood, CA': (34.09, -118.3617), 'Oakland, CA': (37.8044, -122.2712),
    'Brooklyn, NY': (40.6782, -73.9442), 'Asbury Park, NJ': (40.2204, -74.0121),
    'Fort Worth, TX': (32.7555, -97.3308), 'Morrison, CO': (39.6536, -105.1911),
    'San Antonio, TX': (29.4241, -98.4936), 'Ferndale, MI': (42.4606, -83.1346),
    'St. Paul, MN': (44.9537, -93.09), 'Cambridge, MA': (42.3736, -71.1097),
    'Norfolk, VA': (36.8508, -76.2859), 'Glendale, AZ': (33.5387, -112.1860),
    'Englewood, CO': (39.6480, -104.9878),
    # Canadian
    'Toronto, ON': (43.6532, -79.3832), 'Vancouver, BC': (49.2827, -123.1207),
    'Montreal, QC': (45.5017, -73.5673), 'Calgary, AB': (51.0447, -114.0719),
}


@dataclass
class ModelMetrics:
    fill_rate_auc: Optional[float] = None
    revenue_r2: Optional[float] = None
    revenue_mae: Optional[float] = None
    capacity_r2: Optional[float] = None
    guarantee_r2: Optional[float] = None
    guarantee_mae: Optional[float] = None
    n_train: int = 0
    n_artists: int = 0


class TourRecommendationModel:
    """
    Cross-artist recommendation model.
    Trains on ALL artists' settlement data.
    Predicts for ANY artist (known or unknown).
    """

    def __init__(self):
        self.fill_model = None        # P(sellout) classifier
        self.revenue_model = None     # E[log(artist_net)] regressor
        self.capacity_model = None    # optimal capacity regressor
        self.guarantee_model = None   # E[log(guarantee)] regressor
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.metrics = ModelMetrics()
        self.feature_importances = {}

    def train(self, feature_store: FeatureStore):
        """Train all four prediction heads on the full training matrix."""
        tm = feature_store.training_matrix.copy()
        target_cols = ['target_fill_rate', 'target_artist_net', 'target_capacity', 'target_guarantee']
        tm = tm.dropna(subset=[f for f in ALL_FEATURES if f in tm.columns] + target_cols)

        X = tm[[f for f in ALL_FEATURES]].fillna(0).values
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # ── Fill Rate Model (binary: will they sell out?) ──
        y_fill = (tm['target_fill_rate'] >= 0.85).astype(int).values
        self.fill_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        self.fill_model.fit(X_scaled, y_fill)

        # ── Revenue Model (log-transformed regression) ──
        y_rev_raw = tm['target_artist_net'].values
        y_rev = np.log1p(np.maximum(y_rev_raw, 0))
        self.revenue_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        self.revenue_model.fit(X_scaled, y_rev)

        # ── Capacity Model (regression) ──
        y_cap = tm['target_capacity'].values
        self.capacity_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        self.capacity_model.fit(X_scaled, y_cap)

        # ── Guarantee Model (log-transformed regression) ──
        y_guar_raw = tm['target_guarantee'].values
        y_guar = np.log1p(np.maximum(y_guar_raw, 0))
        self.guarantee_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=8, random_state=42)
        self.guarantee_model.fit(X_scaled, y_guar)

        self.is_fitted = True

        # ── Evaluate with artist-level cross-validation ──
        self._evaluate(tm, X_scaled, y_fill, y_rev, y_cap, y_guar)

        # Feature importances (averaged across all 4 models)
        for feat, imp_f, imp_r, imp_c, imp_g in zip(
            ALL_FEATURES,
            self.fill_model.feature_importances_,
            self.revenue_model.feature_importances_,
            self.capacity_model.feature_importances_,
            self.guarantee_model.feature_importances_,
        ):
            self.feature_importances[feat] = round(float((imp_f + imp_r + imp_c + imp_g) / 4), 4)

        print(f"Model trained: {self.metrics.n_train} shows, {self.metrics.n_artists} artists, "
              f"{len(ALL_FEATURES)} features")
        print(f"  Fill AUC: {self.metrics.fill_rate_auc}")
        print(f"  Revenue R2: {self.metrics.revenue_r2}, MAE: ${self.metrics.revenue_mae:,.0f}")
        print(f"  Capacity R2: {self.metrics.capacity_r2}")
        print(f"  Guarantee R2: {self.metrics.guarantee_r2}, MAE: ${self.metrics.guarantee_mae:,.0f}")

        return self

    def _evaluate(self, tm, X_scaled, y_fill, y_rev, y_cap, y_guar):
        """Artist-level cross-validation (hold out entire artists)."""
        artists = tm['artist'].values
        unique_artists = tm['artist'].unique()
        self.metrics.n_train = len(tm)
        self.metrics.n_artists = len(unique_artists)

        if len(unique_artists) < 2:
            self.metrics.fill_rate_auc = round(roc_auc_score(
                y_fill, self.fill_model.predict_proba(X_scaled)[:, 1]), 3) if len(np.unique(y_fill)) > 1 else None
            rev_preds = np.expm1(self.revenue_model.predict(X_scaled))
            rev_actual = np.expm1(y_rev)
            self.metrics.revenue_r2 = round(r2_score(rev_actual, rev_preds), 3)
            self.metrics.revenue_mae = round(mean_absolute_error(rev_actual, rev_preds), 0)
            self.metrics.capacity_r2 = round(r2_score(y_cap, self.capacity_model.predict(X_scaled)), 3)
            guar_preds = np.expm1(self.guarantee_model.predict(X_scaled))
            guar_actual = np.expm1(y_guar)
            self.metrics.guarantee_r2 = round(r2_score(guar_actual, guar_preds), 3)
            self.metrics.guarantee_mae = round(mean_absolute_error(guar_actual, guar_preds), 0)
            return

        # GroupKFold by artist
        gkf = GroupKFold(n_splits=min(len(unique_artists), 5))
        fill_aucs, rev_r2s, rev_maes, cap_r2s, guar_r2s, guar_maes = [], [], [], [], [], []

        hp = dict(n_estimators=150, max_depth=3, subsample=0.8,
                  min_samples_leaf=8, random_state=42)

        for train_idx, test_idx in gkf.split(X_scaled, groups=artists):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]

            # Fill
            y_f_tr, y_f_te = y_fill[train_idx], y_fill[test_idx]
            fm = GradientBoostingClassifier(**hp)
            fm.fit(X_tr, y_f_tr)
            if len(np.unique(y_f_te)) > 1:
                fill_aucs.append(roc_auc_score(y_f_te, fm.predict_proba(X_te)[:, 1]))

            # Revenue (log-space)
            y_r_tr, y_r_te = y_rev[train_idx], y_rev[test_idx]
            rm = GradientBoostingRegressor(**hp)
            rm.fit(X_tr, y_r_tr)
            preds_log = rm.predict(X_te)
            preds_real = np.expm1(preds_log)
            actual_real = np.expm1(y_r_te)
            rev_r2s.append(r2_score(actual_real, preds_real))
            rev_maes.append(mean_absolute_error(actual_real, preds_real))

            # Capacity
            y_c_tr, y_c_te = y_cap[train_idx], y_cap[test_idx]
            cm = GradientBoostingRegressor(**hp)
            cm.fit(X_tr, y_c_tr)
            cap_r2s.append(r2_score(y_c_te, cm.predict(X_te)))

            # Guarantee (log-space)
            y_g_tr, y_g_te = y_guar[train_idx], y_guar[test_idx]
            gm = GradientBoostingRegressor(**hp)
            gm.fit(X_tr, y_g_tr)
            gpreds_log = gm.predict(X_te)
            gpreds_real = np.expm1(gpreds_log)
            gactual_real = np.expm1(y_g_te)
            guar_r2s.append(r2_score(gactual_real, gpreds_real))
            guar_maes.append(mean_absolute_error(gactual_real, gpreds_real))

        self.metrics.fill_rate_auc = round(np.mean(fill_aucs), 3) if fill_aucs else None
        self.metrics.revenue_r2 = round(np.mean(rev_r2s), 3)
        self.metrics.revenue_mae = round(np.mean(rev_maes), 0)
        self.metrics.capacity_r2 = round(np.mean(cap_r2s), 3)
        self.metrics.guarantee_r2 = round(np.mean(guar_r2s), 3)
        self.metrics.guarantee_mae = round(np.mean(guar_maes), 0)

    def predict_for_markets(
        self,
        artist_profile: dict,
        feature_store: FeatureStore,
        markets: Optional[list[str]] = None,
        show_month: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Predict fill probability, revenue, capacity, and guarantee for an artist
        across markets. Works for known artists (from settlement data) AND unknown
        artists (profile built from Chartmetric/Spotify).

        show_month: optional month (1-12) for seasonality-aware predictions.
                    If None, defaults to 6 (June) as neutral baseline.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call .train() first.")

        if markets is None:
            markets = feature_store.get_all_markets()

        market_profiles = feature_store.market_profiles
        am_history = feature_store.artist_market_history
        artist_name = artist_profile.get('artist', 'unknown')

        # Resolve show context defaults from artist profile
        month = show_month if show_month else 6
        quarter = (month - 1) // 3 + 1
        is_peak = 1 if month in (3, 4, 5, 9, 10, 11) else 0

        rows = []
        for mkt in markets:
            mp = market_profiles[market_profiles['market'] == mkt]
            if len(mp) == 0:
                continue
            mp = mp.iloc[0]

            # Artist x Market history
            am = am_history[
                (am_history['artist'] == artist_name) & (am_history['market'] == mkt)
            ]
            has_history = len(am) > 0
            am_row = am.iloc[0] if has_history else {}

            feature_row = {
                # Artist features
                'a_growth_phase': artist_profile.get('growth_phase_num', 0),
                'a_avg_capacity': artist_profile.get('current_avg_capacity', 0),
                'a_avg_guarantee': artist_profile.get('current_avg_guarantee', 0),
                'a_avg_fill_rate': artist_profile.get('current_avg_fill_rate', 0),
                'a_avg_ticket_price': artist_profile.get('current_avg_ticket_price', 0),
                'a_avg_artist_net': artist_profile.get('current_avg_artist_net', 0),
                'a_total_shows': artist_profile.get('total_shows', 0),
                'a_headline_shows': artist_profile.get('total_headline_shows', 0),
                'a_n_markets': artist_profile.get('n_markets_played', 0),
                'a_months_active': artist_profile.get('months_active', 1),
                'a_sellout_rate': artist_profile.get('sellout_rate', 0),
                'a_avg_revenue_per_head': artist_profile.get('avg_revenue_per_head', 0),

                # Artist x Market features
                'am_prior_visits': am_row.get('prior_visits', 0),
                'am_prior_headline_visits': am_row.get('prior_headline_visits', 0),
                'am_prior_avg_fill': am_row.get('prior_avg_fill_rate', 0),
                'am_prior_avg_capacity': am_row.get('prior_avg_capacity', 0),
                'am_prior_last_capacity': am_row.get('prior_last_capacity', 0),
                'am_prior_avg_guarantee': am_row.get('prior_avg_guarantee', 0),
                'am_days_since_last': am_row.get('days_since_last_visit', 9999),
                'am_prior_sellout_rate': am_row.get('prior_sellout_rate', 0),
                'am_has_history': 1 if has_history else 0,

                # Market features
                'market_avg_fill_rate': mp.get('market_avg_fill_rate', 0),
                'market_avg_capacity': mp.get('market_avg_capacity', 0),
                'market_median_capacity': mp.get('market_median_capacity', 0),
                'market_avg_guarantee': mp.get('market_avg_guarantee', 0),
                'market_median_guarantee': mp.get('market_median_guarantee', 0),
                'market_avg_ticket_price': mp.get('market_avg_ticket_price', 0),
                'market_avg_artist_net': mp.get('market_avg_artist_net', 0),
                'market_size_tier_num': mp.get('market_size_tier_num', 0),
                'n_artists_played': mp.get('n_artists_played', 0),
                'n_total_shows': mp.get('n_total_shows', 0),

                # Deal features (use artist's historical averages at prediction time)
                'deal_is_flat': artist_profile.get('pct_flat_deals', 0),
                'deal_is_vs_net': artist_profile.get('pct_vs_net_deals', 0),
                'deal_artist_pct': artist_profile.get('avg_artist_pct', 0),
                'deal_split_point_ratio': (
                    artist_profile.get('avg_split_point', 0) /
                    max(artist_profile.get('current_avg_guarantee', 1), 1)
                ),
                'deal_has_upside': 1.0 - artist_profile.get('pct_flat_deals', 0),

                # Show context features
                'show_month': month,
                'show_quarter': quarter,
                'show_is_weekend': 1,
                'show_is_peak_season': is_peak,
                'show_n_tiers': artist_profile.get('avg_n_tiers', mp.get('market_avg_n_tiers', 3)),
                'show_avg_ticket_price': artist_profile.get('current_avg_ticket_price', mp.get('market_avg_ticket_price', 0)),
                'show_max_ticket_price': artist_profile.get('avg_max_ticket_price', mp.get('market_avg_max_ticket_price', 0)),
                'show_is_21_plus': mp.get('market_pct_21_plus', 0),
            }

            X = np.array([[feature_row[f] for f in ALL_FEATURES]])
            X_scaled = self.scaler.transform(X)

            pred_fill_prob = self.fill_model.predict_proba(X_scaled)[0][1]
            pred_revenue_log = self.revenue_model.predict(X_scaled)[0]
            pred_revenue = max(0, float(np.expm1(pred_revenue_log)))
            pred_capacity = max(50, self.capacity_model.predict(X_scaled)[0])
            pred_guarantee_log = self.guarantee_model.predict(X_scaled)[0]
            pred_guarantee = max(0, float(np.expm1(pred_guarantee_log)))

            rows.append({
                'market': mkt,
                'city': mp.get('city', ''),
                'state': mp.get('state', ''),
                'country': mp.get('country', 'United States'),
                'pred_fill_probability': pred_fill_prob,
                'pred_artist_net': pred_revenue,
                'pred_capacity': pred_capacity,
                'pred_guarantee': pred_guarantee,
                'has_artist_history': has_history,
                'prior_visits': am_row.get('prior_visits', 0),
                'prior_avg_fill': am_row.get('prior_avg_fill_rate', 0),
                'prior_last_capacity': am_row.get('prior_last_capacity', 0),
                'prior_last_guarantee': am_row.get('prior_last_guarantee', 0),
                'days_since_last': am_row.get('days_since_last_visit', 9999),
                'market_n_artists': mp.get('n_artists_played', 0),
                # Market deal context (for tool.py profitability calc)
                'market_pct_vs_net': mp.get('market_pct_vs_net', 0),
                'market_avg_artist_pct': mp.get('market_avg_artist_pct', 0),
                'market_avg_split_point': mp.get('market_avg_split_point', 0),
                'market_avg_max_ticket_price': mp.get('market_avg_max_ticket_price', 0),
                'market_pct_21_plus': mp.get('market_pct_21_plus', 0),
                'market_peak_season_fill': mp.get('market_peak_season_fill', 0),
                'market_off_season_fill': mp.get('market_off_season_fill', 0),
            })

        return pd.DataFrame(rows)

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'TourRecommendationModel':
        return joblib.load(path)


# ── Routing ─────────────────────────────────────────────────────────────

def haversine(c1: tuple, c2: tuple) -> float:
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 3959 * 2 * math.asin(math.sqrt(a))


def get_coords(market: str) -> tuple:
    if market in MARKET_COORDS:
        return MARKET_COORDS[market]
    for key, c in MARKET_COORDS.items():
        if market.split(',')[0] in key:
            return c
    return (39.8283, -98.5795)


def optimize_route(markets: list[str], start_city: Optional[str] = None) -> list[int]:
    """Nearest-neighbor + 2-opt TSP."""
    n = len(markets)
    if n <= 2:
        return list(range(n))
    coords = [get_coords(m) for m in markets]
    dist = np.array([[haversine(coords[i], coords[j]) for j in range(n)] for i in range(n)])

    start = 0
    if start_city:
        for i, m in enumerate(markets):
            if start_city.lower() in m.lower():
                start = i
                break

    visited = [start]
    remaining = set(range(n)) - {start}
    while remaining:
        cur = visited[-1]
        nxt = min(remaining, key=lambda x: dist[cur][x])
        visited.append(nxt)
        remaining.remove(nxt)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                d_old = dist[visited[i-1]][visited[i]] + dist[visited[j]][visited[(j+1) % n]]
                d_new = dist[visited[i-1]][visited[j]] + dist[visited[i]][visited[(j+1) % n]]
                if d_new < d_old:
                    visited[i:j+1] = reversed(visited[i:j+1])
                    improved = True

    total_dist = sum(dist[visited[i]][visited[i+1]] for i in range(n-1))
    return visited, round(total_dist)
