"""
tour_recommender/data.py
Cross-artist data pipeline: parse settlement CSVs → unified feature store.
Handles hundreds of CSVs and 100K+ shows.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


def parse_venue_location(venue_str: str) -> dict:
    lines = str(venue_str).replace('\r', '').split('\n')
    lines = [l.strip() for l in lines if l.strip()]
    result = {'venue_name': lines[0] if lines else None,
              'city': None, 'state': None, 'country': None}
    for line in lines:
        if line in ('United States', 'Canada', 'United Kingdom',
                    'Australia', 'Germany', 'France', 'Japan', 'Mexico',
                    'Netherlands', 'Spain', 'Italy', 'Brazil', 'Sweden',
                    'Norway', 'Denmark', 'Ireland', 'Belgium', 'Switzerland'):
            result['country'] = line
        elif re.search(r'[A-Z]{2}\s+\w\d\w\s*\d\w\d', line):
            m = re.match(r'^(.+?)\s+([A-Z]{2})\s+\w\d\w', line)
            if m:
                result['city'] = m.group(1).strip()
                result['state'] = m.group(2)
        else:
            m = re.match(r'^(.+?)\s+([A-Z]{2})\s*(\d{5})?$', line)
            if m:
                result['city'] = m.group(1).strip()
                result['state'] = m.group(2)
    return result


def parse_scaling_list(scaling_str: str) -> dict:
    if pd.isna(scaling_str):
        return {'avg_ticket_price': 0.0, 'n_tiers': 0, 'max_ticket_price': 0.0,
                'min_ticket_price': 0.0, 'price_spread': 0.0, 'total_scaling_capacity': 0}
    prices = [float(p.replace(',', ''))
              for p in re.findall(r'PRICE:\s*(?:CAD\s*)?\$\s*([\d,.]+)', str(scaling_str))]
    qtys = [int(q) for q in re.findall(r'QTY:\s*(\d+)', str(scaling_str))]
    total_qty = sum(qtys) if qtys else 0
    if total_qty > 0 and len(prices) == len(qtys):
        avg_price = sum(p * q for p, q in zip(prices, qtys)) / total_qty
    elif prices:
        avg_price = float(np.mean(prices))
    else:
        avg_price = 0.0
    return {
        'avg_ticket_price': avg_price,
        'n_tiers': len(prices),
        'max_ticket_price': max(prices) if prices else 0.0,
        'min_ticket_price': min(p for p in prices if p > 0) if any(p > 0 for p in prices) else 0.0,
        'price_spread': (max(prices) - min(p for p in prices if p > 0)) if len([p for p in prices if p > 0]) > 1 else 0.0,
        'total_scaling_capacity': total_qty,
    }


def parse_merchandise(merch_str) -> dict:
    defaults = {'merch_soft_pct': 0.0, 'merch_hard_pct': 0.0,
                'merch_seller_artist': 0, 'merch_seller_venue': 0, 'has_merch_terms': 0}
    if pd.isna(merch_str) or not str(merch_str).strip():
        return defaults
    ms = str(merch_str)
    soft = re.findall(r'(\d+(?:\.\d+)?)\s*%?\s*[/-]?\s*\d*\s*%?\s*[Ss]oft', ms)
    if not soft:
        soft = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*(?:ALL|all)', ms)
    hard = re.findall(r'(\d+(?:\.\d+)?)\s*%?\s*[/-]?\s*\d*\s*%?\s*(?:[Hh]ard|[Rr]ec)', ms)
    seller_artist = bool(re.search(r'[Aa]rtist\s*[Ss]ell|[Ss]eller:\s*ARTIST', ms))
    seller_venue = bool(re.search(r'[Vv]enue\s*[Ss]ell|[Ss]eller:\s*VENUE|FESTIVAL\s*SELL', ms))
    has_100_artist = bool(re.search(r'100%\s*(?:all|ALL).*(?:[Aa]rtist|to\s+Artist)', ms))
    return {
        'merch_soft_pct': float(soft[0]) / 100 if soft else (1.0 if has_100_artist else 0.0),
        'merch_hard_pct': float(hard[0]) / 100 if hard else (1.0 if has_100_artist else 0.0),
        'merch_seller_artist': 1 if (seller_artist or has_100_artist) else 0,
        'merch_seller_venue': 1 if seller_venue else 0,
        'has_merch_terms': 1 if (soft or hard or has_100_artist or seller_artist or seller_venue) else 0,
    }


def parse_age_restriction(age_str) -> int:
    if pd.isna(age_str):
        return 0
    s = str(age_str)
    if 'All Ages' in s or 'AA' in s:
        return 0
    m = re.search(r'(\d+)\+', s)
    return int(m.group(1)) if m else 0


def load_settlement_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding='latin-1')
    date_parts = df['All Dates'].str.extract(r'(\d{2})\.(\d{2})\.(\d{4})')
    df['date'] = pd.to_datetime(
        date_parts[0] + '-' + date_parts[1] + '-' + date_parts[2],
        format='%m-%d-%Y', errors='coerce')
    loc = pd.DataFrame(df['Venue'].apply(parse_venue_location).tolist())
    df = pd.concat([df, loc], axis=1)
    sc = pd.DataFrame(df['Scaling List'].apply(parse_scaling_list).tolist())
    df = pd.concat([df, sc], axis=1)

    # Core numerics
    df['guarantee'] = pd.to_numeric(df['Guarantee'], errors='coerce').fillna(0)
    df['capacity'] = pd.to_numeric(df['Show Capacity'], errors='coerce').fillna(0)
    df['sold'] = pd.to_numeric(df['Settlement Sold Tix'], errors='coerce').fillna(0)
    df['artist_net'] = pd.to_numeric(df['Settlement Artist Net'], errors='coerce').fillna(0)

    # Fill rate with fallback to reported attendance %
    reported_fill = pd.to_numeric(
        df['Final % Of Paid Attendance'].astype(str).str.rstrip('%'),
        errors='coerce').fillna(0) / 100.0
    df['fill_rate'] = np.where(
        (df['sold'] > 0) & (df['capacity'] > 0),
        df['sold'] / df['capacity'],
        np.where(reported_fill > 0, reported_fill, 0))
    df['fill_rate'] = df['fill_rate'].clip(0, 1.5)

    df['revenue_per_head'] = np.where(df['sold'] > 0, df['artist_net'] / df['sold'], 0)

    # Billing type
    df['is_headline'] = df['Billing'].str.contains('Headline', na=False).astype(int)
    df['is_festival'] = df['Billing'].str.contains('Festival', na=False).astype(int)
    df['is_support'] = df['Billing'].str.contains('Support|Special Guest', na=False).astype(int)

    # Deal structure
    deal_col = 'Deal Type' if 'Deal Type' in df.columns else None
    if deal_col:
        df['deal_type_raw'] = df[deal_col].fillna('Unknown')
        df['deal_flat'] = df['deal_type_raw'].str.contains('Flat Guarantee', na=False).astype(int)
        df['deal_vs_net'] = df['deal_type_raw'].str.contains('Vs.*Net', na=False).astype(int)
        df['deal_plus'] = df['deal_type_raw'].str.contains('Plus', na=False).astype(int)
        df['deal_door'] = df['deal_type_raw'].str.contains('Door', na=False).astype(int)
        df['has_upside'] = (df['deal_flat'] == 0).astype(int)
    else:
        for col in ['deal_type_raw', 'deal_flat', 'deal_vs_net', 'deal_plus', 'deal_door', 'has_upside']:
            df[col] = 0
        df['deal_type_raw'] = 'Unknown'

    # Artist % and split point
    if 'Artist %' in df.columns:
        df['artist_pct'] = pd.to_numeric(
            df['Artist %'].astype(str).str.rstrip('%'), errors='coerce').fillna(0) / 100.0
    else:
        df['artist_pct'] = 0.0
    if 'Split Point' in df.columns:
        df['split_point'] = pd.to_numeric(df['Split Point'], errors='coerce').fillna(0)
    else:
        df['split_point'] = 0.0

    # Merchandise
    if 'Merchandise' in df.columns:
        merch = pd.DataFrame(df['Merchandise'].apply(parse_merchandise).tolist())
        df = pd.concat([df, merch], axis=1)
    else:
        for col in ['merch_soft_pct', 'merch_hard_pct', 'merch_seller_artist', 'merch_seller_venue', 'has_merch_terms']:
            df[col] = 0

    # Age restriction
    if 'Age Restriction' in df.columns:
        df['min_age'] = df['Age Restriction'].apply(parse_age_restriction)
    else:
        df['min_age'] = 0
    df['is_21_plus'] = (df['min_age'] >= 21).astype(int)
    df['is_18_plus'] = (df['min_age'] >= 18).astype(int)

    # Seasonality
    df['month'] = df['date'].dt.month.fillna(0).astype(int)
    df['quarter'] = df['date'].dt.quarter.fillna(0).astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek.fillna(0).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype(int)
    df['is_peak_season'] = df['month'].isin([9, 10, 11, 3, 4, 5]).astype(int)

    df['market'] = df['city'].fillna('Unknown') + ', ' + df['state'].fillna('??')
    df['artist'] = df['Artist'].iloc[0] if 'Artist' in df.columns else Path(filepath).stem
    return df


def load_all_csvs(csv_paths: list[str] = None, csv_dir: str = None) -> pd.DataFrame:
    dfs = []
    if csv_paths:
        paths = [Path(p) for p in csv_paths]
    elif csv_dir:
        paths = list(Path(csv_dir).glob('*.csv'))
    else:
        raise ValueError("Provide csv_dir or csv_paths")
    for p in paths:
        try:
            df = load_settlement_csv(str(p))
            df['source_file'] = p.name
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to parse {p.name}: {e}")
    all_shows = pd.concat(dfs, ignore_index=True)
    all_shows = all_shows.dropna(subset=['city', 'date'])
    all_shows = all_shows[all_shows['city'] != 'Unknown']
    print(f"Loaded {len(all_shows):,} shows across {all_shows['artist'].nunique()} artists "
          f"in {all_shows['market'].nunique()} markets")
    return all_shows


# ── Feature Engineering ─────────────────────────────────────────────────


def build_artist_profiles(all_shows: pd.DataFrame) -> pd.DataFrame:
    """One profile row per artist from their most recent settled data."""
    profiles = []
    for artist, group in all_shows.groupby('artist'):
        headline = group[group['is_headline'] == 1].sort_values('date')
        settled = headline[(headline['guarantee'] > 0) | (headline['artist_net'] > 0)]
        if len(settled) == 0:
            settled = headline
        recent = settled.tail(20)

        if len(settled) >= 3:
            yearly = settled.groupby(settled['date'].dt.year)['guarantee'].mean()
            if len(yearly) >= 2 and yearly.iloc[0] > 0:
                cagr = (yearly.iloc[-1] / yearly.iloc[0]) ** (1 / (len(yearly) - 1)) - 1
            else:
                cagr = 0
        else:
            cagr = 0

        avg_guar = recent['guarantee'].mean()
        if avg_guar > 200000: phase, phase_num = 'arena', 4
        elif avg_guar > 50000: phase, phase_num = 'theater', 3
        elif avg_guar > 10000: phase, phase_num = 'club_to_theater', 2
        elif avg_guar > 2000: phase, phase_num = 'club', 1
        else: phase, phase_num = 'emerging', 0

        profiles.append({
            'artist': artist,
            'growth_phase': phase,
            'growth_phase_num': phase_num,
            'current_avg_capacity': recent['capacity'].mean(),
            'current_avg_guarantee': avg_guar,
            'current_avg_fill_rate': recent['fill_rate'].mean(),
            'current_avg_ticket_price': recent['avg_ticket_price'].mean(),
            'current_avg_artist_net': recent['artist_net'].mean(),
            'guarantee_cagr': cagr,
            'total_shows': len(group),
            'total_headline_shows': len(headline),
            'n_markets_played': group['market'].nunique(),
            'months_active': max((group['date'].max() - group['date'].min()).days / 30, 1),
            'headline_ratio': len(headline) / max(len(group), 1),
            'sellout_rate': (headline['fill_rate'] >= 0.95).mean() if len(headline) > 0 else 0,
            # Deal structure
            'avg_artist_pct': recent['artist_pct'].mean() if 'artist_pct' in recent else 0,
            'pct_vs_net_deals': (recent['deal_vs_net'] == 1).mean() if 'deal_vs_net' in recent else 0,
            'pct_flat_deals': (recent['deal_flat'] == 1).mean() if 'deal_flat' in recent else 0,
            'upside_capture_rate': (recent['artist_net'] > recent['guarantee']).mean() if len(recent) > 0 else 0,
            'avg_split_point': recent.loc[recent['split_point'] > 0, 'split_point'].mean() if (recent['split_point'] > 0).any() else 0,
            # Merchandise
            'avg_merch_soft_pct': recent['merch_soft_pct'].mean() if 'merch_soft_pct' in recent else 0,
            'merch_seller_artist_rate': recent['merch_seller_artist'].mean() if 'merch_seller_artist' in recent else 0,
            # Ticket pricing
            'avg_n_tiers': recent['n_tiers'].mean() if 'n_tiers' in recent else 0,
            'avg_max_ticket_price': recent['max_ticket_price'].mean() if 'max_ticket_price' in recent else 0,
            'avg_price_spread': recent['price_spread'].mean() if 'price_spread' in recent else 0,
            # Seasonality
            'peak_season_fill_rate': headline[headline['is_peak_season'] == 1]['fill_rate'].mean() if (headline['is_peak_season'] == 1).any() else 0,
            'off_season_fill_rate': headline[headline['is_peak_season'] == 0]['fill_rate'].mean() if (headline['is_peak_season'] == 0).any() else 0,
            # Revenue
            'avg_revenue_per_head': recent.loc[recent['revenue_per_head'] > 0, 'revenue_per_head'].mean() if (recent['revenue_per_head'] > 0).any() else 0,
        })
    return pd.DataFrame(profiles)


def build_market_profiles(all_shows: pd.DataFrame) -> pd.DataFrame:
    """One profile row per market aggregated across ALL artists."""
    profiles = []
    for market, group in all_shows.groupby('market'):
        headline = group[group['is_headline'] == 1]
        n_artists = group['artist'].nunique()
        avg_cap = headline['capacity'].mean() if len(headline) > 0 else 0
        if avg_cap > 10000: tier_num = 3
        elif avg_cap > 2000: tier_num = 2
        elif avg_cap > 500: tier_num = 1
        else: tier_num = 0

        profiles.append({
            'market': market,
            'city': group['city'].iloc[0],
            'state': group['state'].iloc[0],
            'country': group['country'].iloc[0],
            'n_artists_played': n_artists,
            'n_total_shows': len(group),
            'n_headline_shows': len(headline),
            'market_avg_fill_rate': headline['fill_rate'].mean() if len(headline) > 0 else 0,
            'market_avg_capacity': avg_cap,
            'market_median_capacity': headline['capacity'].median() if len(headline) > 0 else 0,
            'market_avg_guarantee': headline['guarantee'].mean() if len(headline) > 0 else 0,
            'market_median_guarantee': headline['guarantee'].median() if len(headline) > 0 else 0,
            'market_avg_ticket_price': group['avg_ticket_price'].mean(),
            'market_avg_artist_net': headline['artist_net'].mean() if len(headline) > 0 else 0,
            'market_size_tier_num': tier_num,
            # Deal structure
            'market_pct_vs_net': headline['has_upside'].mean() if len(headline) > 0 and 'has_upside' in headline else 0,
            'market_avg_artist_pct': headline['artist_pct'].mean() if len(headline) > 0 and 'artist_pct' in headline else 0,
            'market_avg_split_point': headline.loc[headline['split_point'] > 0, 'split_point'].mean() if len(headline) > 0 and 'split_point' in headline and (headline['split_point'] > 0).any() else 0,
            # Ticket pricing
            'market_avg_n_tiers': group['n_tiers'].mean() if 'n_tiers' in group else 0,
            'market_avg_max_ticket_price': group['max_ticket_price'].mean() if 'max_ticket_price' in group else 0,
            # Seasonality
            'market_peak_season_fill': headline[headline['is_peak_season'] == 1]['fill_rate'].mean() if len(headline) > 0 and 'is_peak_season' in headline and (headline['is_peak_season'] == 1).any() else 0,
            'market_off_season_fill': headline[headline['is_peak_season'] == 0]['fill_rate'].mean() if len(headline) > 0 and 'is_peak_season' in headline and (headline['is_peak_season'] == 0).any() else 0,
            # Age
            'market_pct_21_plus': (group['is_21_plus'] == 1).mean() if 'is_21_plus' in group else 0,
        })
    return pd.DataFrame(profiles)


def build_artist_market_history(all_shows: pd.DataFrame) -> pd.DataFrame:
    """One row per (artist, market) pair with interaction history."""
    now = pd.Timestamp.now()
    interactions = []
    for (artist, market), group in all_shows.groupby(['artist', 'market']):
        headline = group[group['is_headline'] == 1].sort_values('date')
        settled = headline[(headline['guarantee'] > 0) | (headline['artist_net'] > 0)]
        interactions.append({
            'artist': artist, 'market': market,
            'prior_visits': len(group),
            'prior_headline_visits': len(headline),
            'prior_avg_fill_rate': group['fill_rate'].mean(),
            'prior_max_fill_rate': group['fill_rate'].max(),
            'prior_avg_capacity': group['capacity'].mean(),
            'prior_max_capacity': group['capacity'].max(),
            'prior_last_capacity': group.nlargest(1, 'date')['capacity'].iloc[0],
            'prior_avg_guarantee': settled['guarantee'].mean() if len(settled) > 0 else 0,
            'prior_last_guarantee': settled.nlargest(1, 'date')['guarantee'].iloc[0] if len(settled) > 0 else 0,
            'prior_avg_artist_net': settled['artist_net'].mean() if len(settled) > 0 else 0,
            'prior_sellout_rate': (group['fill_rate'] >= 0.95).mean(),
            'days_since_last_visit': (now - group['date'].max()).days,
        })
    return pd.DataFrame(interactions)


def build_training_matrix(all_shows: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full training matrix: one row per headline show with
    artist features (rolling, no leakage) + market features + targets.
    """
    all_shows = all_shows.sort_values('date').reset_index(drop=True)
    rows = []

    for artist, artist_shows in all_shows.groupby('artist'):
        artist_shows = artist_shows.sort_values('date').reset_index(drop=True)

        for i in range(len(artist_shows)):
            show = artist_shows.iloc[i]
            if show['is_headline'] != 1:
                continue
            if show['guarantee'] == 0 and show['artist_net'] == 0:
                continue
            if show['capacity'] == 0:
                continue

            prior = artist_shows.iloc[:i]
            prior_headline = prior[prior['is_headline'] == 1]
            prior_settled = prior_headline[
                (prior_headline['guarantee'] > 0) | (prior_headline['artist_net'] > 0)]
            if len(prior_headline) < 1:
                continue

            recent = prior_settled.tail(20) if len(prior_settled) > 0 else prior_headline.tail(20)
            avg_guar = recent['guarantee'].mean()
            if avg_guar > 200000: phase_num = 4
            elif avg_guar > 50000: phase_num = 3
            elif avg_guar > 10000: phase_num = 2
            elif avg_guar > 2000: phase_num = 1
            else: phase_num = 0

            prior_market = prior[prior['market'] == show['market']]
            prior_h_market = prior_market[prior_market['is_headline'] == 1]

            rows.append({
                # Artist features
                'a_growth_phase': phase_num,
                'a_avg_capacity': recent['capacity'].mean(),
                'a_avg_guarantee': avg_guar,
                'a_avg_fill_rate': recent['fill_rate'].mean(),
                'a_avg_ticket_price': recent['avg_ticket_price'].mean(),
                'a_avg_artist_net': recent['artist_net'].mean(),
                'a_total_shows': len(prior),
                'a_headline_shows': len(prior_headline),
                'a_n_markets': prior['market'].nunique(),
                'a_months_active': max((prior['date'].max() - prior['date'].min()).days / 30, 0.1),
                'a_sellout_rate': (prior_headline['fill_rate'] >= 0.95).mean(),
                'a_avg_revenue_per_head': recent.loc[recent['revenue_per_head'] > 0, 'revenue_per_head'].mean() if (recent['revenue_per_head'] > 0).any() else 0,

                # Artist-market features
                'am_prior_visits': len(prior_market),
                'am_prior_headline_visits': len(prior_h_market),
                'am_prior_avg_fill': prior_market['fill_rate'].mean() if len(prior_market) > 0 else 0,
                'am_prior_avg_capacity': prior_market['capacity'].mean() if len(prior_market) > 0 else 0,
                'am_prior_last_capacity': prior_market.nlargest(1, 'date')['capacity'].iloc[0] if len(prior_market) > 0 else 0,
                'am_prior_avg_guarantee': prior_h_market['guarantee'].mean() if len(prior_h_market) > 0 else 0,
                'am_days_since_last': (show['date'] - prior_market['date'].max()).days if len(prior_market) > 0 else 9999,
                'am_prior_sellout_rate': (prior_market['fill_rate'] >= 0.95).mean() if len(prior_market) > 0 else 0,
                'am_has_history': 1 if len(prior_market) > 0 else 0,

                # Deal features (from this show)
                'deal_is_flat': show.get('deal_flat', 0),
                'deal_is_vs_net': show.get('deal_vs_net', 0),
                'deal_artist_pct': show.get('artist_pct', 0),
                'deal_split_point_ratio': show['split_point'] / max(show['guarantee'], 1) if show.get('split_point', 0) > 0 else 0,
                'deal_has_upside': show.get('has_upside', 0),

                # Show context features
                'show_month': show.get('month', 0),
                'show_quarter': show.get('quarter', 0),
                'show_is_weekend': show.get('is_weekend', 0),
                'show_is_peak_season': show.get('is_peak_season', 0),
                'show_n_tiers': show.get('n_tiers', 0),
                'show_avg_ticket_price': show.get('avg_ticket_price', 0),
                'show_max_ticket_price': show.get('max_ticket_price', 0),
                'show_is_21_plus': show.get('is_21_plus', 0),

                # Metadata
                'market': show['market'],
                'artist': artist,
                'date': show['date'],

                # Targets
                'target_fill_rate': show['fill_rate'],
                'target_artist_net': show['artist_net'],
                'target_capacity': show['capacity'],
                'target_guarantee': show['guarantee'],
            })

    training_df = pd.DataFrame(rows)
    print(f"Built training matrix: {len(training_df):,} rows, "
          f"{training_df['artist'].nunique()} artists, "
          f"{training_df['market'].nunique()} markets")
    return training_df


@dataclass
class FeatureStore:
    """Central store for all computed features."""
    all_shows: pd.DataFrame
    artist_profiles: pd.DataFrame
    market_profiles: pd.DataFrame
    artist_market_history: pd.DataFrame
    training_matrix: pd.DataFrame

    @classmethod
    def build(cls, csv_paths: list[str] = None, csv_dir: str = None) -> 'FeatureStore':
        all_shows = load_all_csvs(csv_dir=csv_dir, csv_paths=csv_paths)
        artist_profiles = build_artist_profiles(all_shows)
        market_profiles = build_market_profiles(all_shows)
        am_history = build_artist_market_history(all_shows)
        training_matrix = build_training_matrix(all_shows)

        # Merge market features into training matrix
        market_cols = [c for c in market_profiles.columns
                       if c.startswith('market_') or c in ('n_artists_played', 'n_total_shows')]
        training_matrix = training_matrix.merge(
            market_profiles[['market'] + market_cols], on='market', how='left')

        return cls(all_shows=all_shows, artist_profiles=artist_profiles,
                   market_profiles=market_profiles, artist_market_history=am_history,
                   training_matrix=training_matrix)

    def get_artist_profile(self, artist_name: str) -> Optional[dict]:
        match = self.artist_profiles[self.artist_profiles['artist'] == artist_name]
        return match.iloc[0].to_dict() if len(match) > 0 else None

    def get_all_markets(self) -> list[str]:
        return self.market_profiles['market'].tolist()
