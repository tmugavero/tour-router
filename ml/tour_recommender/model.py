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
    # Core US cities (original)
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
    'West Hollywood, CA': (34.0900, -118.3617), 'Oakland, CA': (37.8044, -122.2712),
    'Brooklyn, NY': (40.6782, -73.9442), 'Asbury Park, NJ': (40.2204, -74.0121),
    'Fort Worth, TX': (32.7555, -97.3308), 'Morrison, CO': (39.6536, -105.1911),
    'San Antonio, TX': (29.4241, -98.4936), 'Ferndale, MI': (42.4606, -83.1346),
    'St. Paul, MN': (44.9537, -93.0900), 'Cambridge, MA': (42.3736, -71.1097),
    'Norfolk, VA': (36.8508, -76.2859), 'Glendale, AZ': (33.5387, -112.1860),
    'Englewood, CO': (39.6480, -104.9878),
    # Expanded US markets
    'Abiquiu, NM': (36.2081, -106.3222),
    'Albuquerque, NM': (35.0853, -106.6056),
    'Amherst, MA': (42.3731, -72.5199),
    'Anaheim, CA': (33.8366, -117.9143),
    'Ann Arbor, MI': (42.2808, -83.7430),
    'Asheville, NC': (35.5951, -82.5515),
    'Aspen, CO': (39.1911, -106.8175),
    'Athens, GA': (33.9519, -83.3576),
    'Bakersfield, CA': (35.3733, -119.0187),
    'Baltimore, MD': (39.2904, -76.6122),
    'Baton Rouge, LA': (30.4515, -91.1871),
    'Beech Mountain, NC': (36.2168, -81.9026),
    'Bellingham, WA': (48.7519, -122.4787),
    'Belmont, NY': (42.2248, -78.0333),
    'Bend, OR': (44.0582, -121.3153),
    'Berkeley, CA': (37.8716, -122.2727),
    'Big Sur, CA': (36.2502, -121.7938),
    'Birmingham, AL': (33.5186, -86.8104),
    'Bloomington, IL': (40.4842, -88.9937),
    'Bloomington, IN': (39.1653, -86.5264),
    'Boise, ID': (43.6150, -116.2023),
    'Boulder, CO': (40.0150, -105.2705),
    'Bridgeview, IL': (41.7414, -87.8031),
    'Bristow, VA': (38.7179, -77.5458),
    'Bronx, NY': (40.8448, -73.8648),
    'Broomfield, CO': (39.9205, -105.0867),
    'Buffalo, NY': (42.8864, -78.8784),
    'Carnation, WA': (47.6476, -121.9160),
    'Caroga Lake, NY': (43.1537, -74.4749),
    'Carrboro, NC': (35.9104, -79.0752),
    'Charleston, SC': (32.7765, -79.9311),
    'Chautauqua, NY': (42.2092, -79.4647),
    'Cheyenne, WY': (41.1400, -104.8202),
    'Cincinnati OH, OH': (39.1031, -84.5120),
    'Cincinnati, OH': (39.1031, -84.5120),
    'Cleveland Heights, OH': (41.5120, -81.5568),
    'Clinton Twp, MI': (42.5870, -82.9193),
    'Columbia, MD': (39.2037, -76.8610),
    'Costa Mesa, CA': (33.6411, -117.9187),
    'Cudahy, WI': (42.9592, -87.8625),
    'Cuyahoga Falls, OH': (41.1334, -81.4846),
    'Dana Point, CA': (33.4669, -117.6981),
    'Del Mar, CA': (32.9595, -117.2653),
    'Denton, TX': (33.2148, -97.1331),
    'Des Moines, IA': (41.5868, -93.6250),
    'Dover, DE': (39.1582, -75.5244),
    'Durham, NC': (35.9940, -78.8986),
    'El Cajon, CA': (32.7948, -116.9625),
    'Eugene, OR': (44.0521, -123.0868),
    'Everett, MA': (42.4084, -71.0537),
    'Ewing, NJ': (40.2618, -74.7968),
    'Fayetteville, AR': (36.0626, -94.1574),
    'Forest Grove, OR': (45.5207, -123.1051),
    'Fort Lauderdale, FL': (26.1224, -80.1373),
    'Franklin, TN': (35.9251, -86.8689),
    'Gonzales, TX': (29.5058, -97.4517),
    'Grand Rapids, MI': (42.9634, -85.6681),
    'Great Barrington, MA': (42.1954, -73.3623),
    'Greenwood Village, CO': (39.6136, -104.9122),
    'Gulf Shores, AL': (30.2460, -87.7008),
    'Hamden, CT': (41.3959, -72.8968),
    'Hanover, NH': (43.7022, -72.2896),
    'Harrisburg, PA': (40.2732, -76.8867),
    'Henderson, NV': (36.0397, -114.9817),
    'Highland, CA': (34.1283, -117.2086),
    'Hollywood, CA': (34.0928, -118.3287),
    'Hollywood, FL': (26.0112, -80.1495),
    'Honolulu, HI': (21.3069, -157.8583),
    'Hudson, NY': (42.2529, -73.7921),
    'Independence, MO': (39.0911, -94.4155),
    'Indio, CA': (33.7206, -116.2156),
    'Inglewood, CA': (33.9617, -118.3531),
    'Iowa City, IA': (41.6611, -91.5302),
    'Irving, TX': (32.8140, -96.9489),
    'Isle of Palms, SC': (32.7872, -79.7731),
    'Jersey City, NJ': (40.7178, -74.0431),
    'Kirksville, MO': (40.1948, -92.5830),
    'La Jolla, CA': (32.8328, -117.2713),
    'Lake Buena Vista, FL': (28.3852, -81.5639),
    'Lakewood, OH': (41.4820, -81.7982),
    'Las Vegas, NV': (36.1699, -115.1398),
    'Lawrence, KS': (38.9717, -95.2353),
    'Lexington, KY': (38.0406, -84.5037),
    'Lincoln, NE': (40.8136, -96.7026),
    'Littleton, CO': (39.6133, -105.0166),
    'Long Beach, CA': (33.7701, -118.1937),
    'Louisville, KY': (38.2527, -85.7585),
    'Madison, WI': (43.0731, -89.4012),
    'Malibu, CA': (34.0259, -118.7798),
    'Manchester, TN': (35.4817, -86.0886),
    'Marfa, TX': (30.3090, -104.0207),
    'Maspeth, NY': (40.7284, -73.9076),
    'Mesa, AZ': (33.4152, -111.8315),
    'Miami Beach, FL': (25.7907, -80.1300),
    'Millvale, PA': (40.4851, -79.9778),
    'Milwaukee, WI': (43.0389, -87.9065),
    'Montauk, NY': (41.0534, -71.9571),
    'Mountain View, CA': (37.3861, -122.0839),
    'Napa, CA': (38.2975, -122.2869),
    'New Kensington, PA': (40.5698, -79.7578),
    'Newport, KY': (39.0920, -84.4947),
    'Newport, RI': (41.4901, -71.3128),
    'Noblesville, IN': (40.0456, -86.0086),
    'Northfield, MN': (44.4583, -93.1616),
    'Oberlin, OH': (41.2934, -82.2174),
    'Ocean City, MD': (38.3365, -75.0849),
    'Ogden, UT': (41.2230, -111.9738),
    'Okeechobee, FL': (27.2436, -80.8320),
    'Oklahoma City, OK': (35.4676, -97.5164),
    'Omaha, NE': (41.2565, -95.9345),
    'Palmer, AK': (61.5993, -149.1128),
    'Pasadena, CA': (34.1478, -118.1445),
    'Paso Robles, CA': (35.6369, -120.6910),
    'Perris, CA': (33.7825, -117.2286),
    'Pioneertown, CA': (34.1731, -116.5197),
    'Pomona, CA': (34.0551, -117.7500),
    'Port Townsend, WA': (48.1173, -122.7594),
    'Portland, ME': (43.6591, -70.2568),
    'Portsmouth, NH': (43.0718, -70.7626),
    'Princeton, NJ': (40.3573, -74.6672),
    'Providence, RI': (41.8240, -71.4128),
    'Queens, NY': (40.7282, -73.7949),
    'Redding, CA': (40.5865, -122.3917),
    'Redondo Beach, CA': (33.8492, -118.3884),
    'Reno, NV': (39.5296, -119.8138),
    'Ridgedale, MO': (36.6098, -93.3101),
    'Riverside, CA': (33.9533, -117.3961),
    'Rochester, NY': (43.1566, -77.6088),
    'Roseville, CA': (38.7521, -121.2880),
    'Rothbury, MI': (43.5137, -86.3492),
    'Royal Oak, MI': (42.4895, -83.1446),
    'Sacramento, CA': (38.5816, -121.4944),
    'San Bernardino, CA': (34.1083, -117.2898),
    'San Jose, CA': (37.3382, -121.8863),
    'San Marcos, CA': (33.1434, -117.1661),
    'San Pedro, CA': (33.7361, -118.2922),
    'Sandy, UT': (40.5649, -111.8389),
    'Santa Barbara, CA': (34.4208, -119.6982),
    'Santa Cruz, CA': (36.9741, -122.0308),
    'Santa Fe, NM': (35.6870, -105.9378),
    'Santa Monica, CA': (34.0195, -118.4912),
    'Saratoga Springs, NY': (43.0831, -73.7846),
    'Saratoga, CA': (37.2638, -122.0231),
    'Saxapahaw, NC': (35.9490, -79.3286),
    'Silver Spring, MD': (38.9907, -77.0261),
    'Snowmass Village, CO': (39.2086, -106.9317),
    'Solana Beach, CA': (32.9912, -117.2713),
    'Somerville, MA': (42.3876, -71.0995),
    'South Burlington, VT': (44.4669, -73.2121),
    'Springfield, MO': (37.2090, -93.2923),
    'St. Augustine, FL': (29.8947, -81.3145),
    'St. Charles, IA': (41.3031, -93.7991),
    'St. Petersburg, FL': (27.7676, -82.6403),
    'Sterling Heights, MI': (42.5803, -83.0302),
    'Syracuse, NY': (43.0481, -76.1474),
    'Tacoma, WA': (47.2529, -122.4443),
    'Tallahassee, FL': (30.4518, -84.2807),
    'Tampa Bay, FL': (27.9506, -82.4572),
    'Tempe, AZ': (33.4255, -111.9400),
    'Tinley Park, IL': (41.5731, -87.7873),
    'Troutdale, OR': (45.5388, -122.3885),
    'Tucson, AZ': (32.2226, -110.9747),
    'Tulsa, OK': (36.1540, -95.9928),
    'Uncasville, CT': (41.4473, -72.1054),
    'Urbana, IL': (40.1106, -88.2073),
    'Vail, CO': (39.6433, -106.3781),
    'Venice, CA': (33.9850, -118.4695),
    'Ventura, CA': (34.2805, -119.2945),
    'Virginia Beach, VA': (36.8529, -75.9780),
    'Waltham, MA': (42.3765, -71.2356),
    'Waterville, ME': (44.5523, -69.6317),
    'West Columbia, SC': (33.9932, -81.0723),
    'West Palm Beach, FL': (26.7153, -80.0534),
    'West Valley City, UT': (40.6916, -112.0010),
    'Wheatland, CA': (39.0088, -121.4238),
    'Williamsburg, VA': (37.2707, -76.7075),
    'Wilmington, DE': (39.7447, -75.5484),
    'Worcester, MA': (42.2626, -71.8023),
    'Yuba City, CA': (39.1404, -121.6169),
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
