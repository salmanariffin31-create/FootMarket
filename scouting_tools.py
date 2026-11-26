"""Scouting views for the existing FootMarket V4 model."""
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
import streamlit as st
from fm_utils import load_player_data, load_player_list, euros, prediction_range

ROOT = Path(__file__).resolve().parent


def _signature():
    paths = [ROOT / p for p in ('data/processed/footmarket_final.csv',
             'models/weighted_log_regression.joblib', 'models/polynomial_preprocessor.joblib')]
    return tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


def _predict_rows(rows, model, preprocessor):
    data = rows.copy()
    data['age_squared'] = data['age'] ** 2
    features = list(preprocessor.feature_names_in_)
    values = model.predict(preprocessor.transform(data[features]))
    return np.maximum(np.expm1(values), 0)


@st.cache_data(ttl=3600, max_entries=2, show_spinner='Preparing scouting rankings…')
def _latest_predictions(signature, _model, _preprocessor):
    df = load_player_data().sort_values('valuation_date')
    latest = df.drop_duplicates('player_id', keep='last').copy()
    latest['estimate'] = _predict_rows(latest, _model, _preprocessor)
    latest = latest[np.isfinite(latest['estimate'])].copy()
    latest['difference'] = latest['estimate'] - latest['market_value']
    latest['gap_pct'] = latest['difference'] / latest['market_value'].where(latest['market_value'] > 0) * 100
    return latest


def _header(kicker, title, subtitle):
    st.html(f'<div class="fm-comparison-header"><div class="fm-eyebrow">{escape(kicker)}</div>'
            f'<div class="fm-hero-title">{escape(title)}</div>'
            f'<div class="fm-hero-description">{escape(subtitle)}</div></div>')


def _filtered(model, preprocessor, clubs, prefix, gems=False):
    df = _latest_predictions(_signature(), model, preprocessor)
    if df.empty:
        st.warning('No eligible valuation records found.')
        return df
    club_names = dict(zip(clubs['club_id'].astype(str), clubs['name']))
    df['club'] = df['last_club_id'].astype(str).map(club_names).fillna(df['last_club_id'].astype(str))
    anchor = df['valuation_date'].max()
    st.caption(f'Latest available record per player. Dataset ends {anchor:%d %b %Y}; not live market data.')
    c1, c2, c3 = st.columns(3)
    with c1:
        position = st.selectbox('Position', ['All'] + sorted(df['position'].dropna().unique().tolist()), key=prefix+'pos')
        club = st.selectbox('Club', ['All'] + sorted(df['club'].unique().tolist()), key=prefix+'club')
    with c2:
        max_age = st.slider('Maximum age', 15, 50, 23 if gems else 50, key=prefix+'age')
        min_minutes = st.number_input('Minimum minutes (previous 365 days)', min_value=90, max_value=10000, value=900 if gems else 90, step=90, key=prefix+'mins')
    with c3:
        recency = st.selectbox('Record recency (relative to dataset end)', ['Last 365 days', 'Last 730 days', 'All dates'], key=prefix+'recent')
        limit = st.selectbox('Results', [10, 25, 50, 100], index=1, key=prefix+'limit')
    df = df[(df['age'] <= max_age) & (df['minutes'] >= min_minutes)].copy()
    if position != 'All': df = df[df['position'] == position]
    if club != 'All': df = df[df['club'] == club]
    if recency != 'All dates':
        df = df[df['valuation_date'] >= anchor - pd.Timedelta(days=int(recency.split()[1]))]
    return df, limit


def _table(df, prefix):
    cols = {'name':'Player', 'club':'Club', 'position':'Position', 'age':'Age', 'valuation_date':'Date',
            'minutes':'Minutes', 'market_value':'Recorded (€)', 'estimate':'Model estimate (€)',
            'difference':'Difference (€)', 'gap_pct':'Difference (%)'}
    table = df[list(cols)].rename(columns=cols).reset_index(drop=True)
    table.index += 1
    table.index.name = 'Rank'
    st.dataframe(table, width='stretch', column_config={
        'Date':st.column_config.DateColumn(format='DD MMM YYYY'),
        'Recorded (€)':st.column_config.NumberColumn(format='€%,.0f'),
        'Model estimate (€)':st.column_config.NumberColumn(format='€%,.0f'),
        'Difference (€)':st.column_config.NumberColumn(format='€%,.0f'),
        'Difference (%)':st.column_config.NumberColumn(format='%.1f%%'),
    })
    st.download_button('Download results (CSV)', table.to_csv(index=True), file_name=f'footmarket_{prefix}.csv', mime='text/csv', key=prefix+'download')


def show_leaderboard(model, preprocessor, clubs):
    _header('MARKET INTELLIGENCE', 'PLAYER RANKINGS.', 'Explore historical player valuations and rank players by the V4 model or recorded reference values.')
    result = _filtered(model, preprocessor, clubs, 'lb_')
    if isinstance(result, pd.DataFrame): return
    df, limit = result
    sort = st.radio('Rank by', ['Model estimate', 'Recorded value'], horizontal=True, key='lb_sort')
    column = 'estimate' if sort == 'Model estimate' else 'market_value'
    st.caption(f'{len(df):,} players match the filters.')
    if df.empty:
        st.info('No players match these filters. Try widening them.'); return
    _table(df.sort_values([column, 'player_id'], ascending=[False, True]).head(limit), 'leaderboard')


def show_hidden_gems(model, preprocessor, clubs):
    _header('SCOUTING INTELLIGENCE', 'HIDDEN GEMS.', 'Find players whose performance-based model estimate exceeds their recorded Transfermarkt valuation.')
    result = _filtered(model, preprocessor, clubs, 'hg_', gems=True)
    if isinstance(result, pd.DataFrame): return
    df, limit = result
    a,b,c = st.columns(3)
    with a:
        minimum = st.number_input('Minimum recorded value (€)', min_value=1, value=100000, step=50000, key='hg_min')
    with b:
        threshold = st.slider('Minimum model premium (%)', 0, 300, 25, key='hg_pct')
    with c:
        ranking = st.selectbox('Rank by', ['Percentage difference', 'Euro difference'], key='hg_sort')
    df = df[(df['market_value'] >= minimum) & (df['difference'] > 0) & (df['gap_pct'] >= threshold)]
    st.caption(f'{len(df):,} potential scouting candidates match your filters.')
    st.warning('A positive valuation gap is a model discrepancy, not proof of an undervalued player, transfer opportunity or investment return. Model errors and missing features can create large gaps.')
    if df.empty:
        st.info('No candidates match. Reduce the minimum premium or widen the filters.'); return
    col = 'gap_pct' if ranking == 'Percentage difference' else 'difference'
    _table(df.sort_values([col,'player_id'], ascending=[False,True]).head(limit), 'hidden_gems')


def show_what_if(model, preprocessor, clubs):
    _header('SIMULATION LAB', 'WHAT IF?', 'Change the selected historical player record and see how the existing model responds.')
    df = load_player_data()
    players = load_player_list()
    options = dict(zip(players['label'], players['player_id']))
    name = st.selectbox('Select player', list(options), index=None, placeholder='Search a player…', key='wi_player')
    if name is None:
        st.info('Select a player to start a simulation.'); return
    pid = options[name]
    history = df[df['player_id'] == pid].sort_values('valuation_date')
    dates = history['valuation_date'].dt.strftime('%Y-%m-%d').unique().tolist()
    date = st.selectbox('Historical valuation date', dates, index=len(dates)-1, key=f'wi_date_{pid}')
    row = history[history['valuation_date'] == pd.Timestamp(date)].iloc[0]
    baseline = float(_predict_rows(pd.DataFrame([row.to_dict()]), model, preprocessor)[0])
    club_names = dict(zip(clubs['club_id'].astype(str), clubs['name']))
    original_club = str(row['last_club_id'])
    club_names.setdefault(original_club, 'Original club '+original_club)
    ids = [original_club] + sorted((k for k in club_names if k != original_club), key=lambda k: club_names[k])
    key = f'wi_{pid}_{date}'
    st.caption(f'Baseline: {row["name"]} · {date} · {row["position"]}. Position stays fixed.')
    st.metric('Baseline model estimate', euros(baseline))
    with st.form(key):
        a,b = st.columns(2)
        with a:
            age = st.number_input('Age', min_value=1, max_value=max(60,int(row['age'])), value=int(row['age']), key=key+'age')
            goals = st.number_input('Goals', min_value=0, max_value=max(150,int(row['goals'])), value=int(row['goals']), key=key+'g')
            assists = st.number_input('Assists', min_value=0, max_value=max(150,int(row['assists'])), value=int(row['assists']), key=key+'a')
        with b:
            apps = st.number_input('Appearances', min_value=1, max_value=max(120,int(row['appearances'])), value=int(row['appearances']), key=key+'apps')
            mins = st.number_input('Minutes played', min_value=90, max_value=max(15000,int(row['minutes'])), value=int(row['minutes']), step=1, key=key+'mins')
            club = st.selectbox('Last recorded club', ids, format_func=lambda k: f'{club_names[k]} ({k})', key=key+'club')
        submitted = st.form_submit_button('Run simulation', width='stretch')
    if not submitted: return
    if mins > apps*120:
        st.error('Minutes cannot exceed 120 per appearance on average.'); return
    modified = row.to_dict()
    modified.update(age=age, goals=goals, assists=assists, appearances=apps, minutes=mins, last_club_id=club,
                    goals_per_90=goals/mins*90, assists_per_90=assists/mins*90, goal_contributions=goals+assists,
                    contributions_per_90=(goals+assists)/mins*90, minutes_per_appearance=mins/apps)
    simulated = float(_predict_rows(pd.DataFrame([modified]), model, preprocessor)[0])
    if not np.isfinite(simulated):
        st.error('This scenario generated an invalid estimate. Try more realistic inputs.'); return
    diff = simulated-baseline
    pct = f'{diff/baseline*100:+.1f}%' if baseline > 0 else None
    a,b,c = st.columns(3)
    a.metric('Baseline estimate', euros(baseline))
    b.metric('Simulated estimate', euros(simulated), delta=pct, delta_color='off')
    c.metric('Change in estimate', ('+' if diff>=0 else '−')+euros(abs(diff)))
    low, high = prediction_range(simulated)
    st.caption(f'Typical validation-based range: {euros(low)} – {euros(high)}. Not a guaranteed prediction interval.')
    st.bar_chart(pd.DataFrame({'Estimated value (€)': [baseline,simulated]}, index=['Baseline','Scenario']))
    st.info('This is a model sensitivity experiment, not a forecast or causal claim. Extreme input combinations may be outside the training distribution. To reset, switch the selected record or edit the fields back to baseline.')
