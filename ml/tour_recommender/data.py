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
        return {'avg_ticket_price': 0.0, 'n_tiers': 0}
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
    return {'avg_ticket_price': avg_price, 'n_tiers': len(prices)}


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
    df['guarantee'] = pd.to_numeric(df['Guarantee'], errors='coerce').fillna(0)
    df['capacity'] = pd.to_numeric(df['Show Capacity'], errors='coerce').fillna(0)
    df['sold'] = pd.to_numeric(df['Settlement Sold Tix'], errors='coerce').fillna(0)
    df['artist_net'] = pd.to_numeric(df['Settlement Artist Net'], errors='coerce').fillna(0)
    df['fill_rate'] = np.where(df['capacity'] > 0, df['sold'] / df['capacity'], 0)
    df['fill_rate'] = df['fill_rate'].clip(0, 2.0)
    df['revenue_per_head'] = np.where(df['sold'] > 0, df['artist_net'] / df['sold'], 0)
    df['is_headline'] = df['Billing'].str.contains('Headline', na=False).astype(int)
    df['is_festival'] = df['Billing'].str.contains('Festival', na=False).astype(int)
    df['is_support'] = df['Billing'].str.contains('Support|Special Guest', na=False).astype(int)
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

                'am_prior_visits': len(prior_market),
                'am_prior_headline_visits': len(prior_h_market),
                'am_prior_avg_fill': prior_market['fill_rate'].mean() if len(prior_market) > 0 else 0,
                'am_prior_avg_capacity': prior_market['capacity'].mean() if len(prior_market) > 0 else 0,
                'am_prior_last_capacity': prior_market.nlargest(1, 'date')['capacity'].iloc[0] if len(prior_market) > 0 else 0,
                'am_prior_avg_guarantee': prior_h_market['guarantee'].mean() if len(prior_h_market) > 0 else 0,
                'am_days_since_last': (show['date'] - prior_market['date'].max()).days if len(prior_market) > 0 else 9999,
                'am_prior_sellout_rate': (prior_market['fill_rate'] >= 0.95).mean() if len(prior_market) > 0 else 0,
                'am_has_history': 1 if len(prior_market) > 0 else 0,

                'market': show['market'],
                'artist': artist,
                'date': show['date'],

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
        market_cols = [c for c in market_profiles.columns if c.startswith('market_') or c in ('n_artists_played', 'n_total_shows')]
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
