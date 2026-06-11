import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ★ 日本語フォント設定（文字化け修正） ★
# ============================================================
import matplotlib
import matplotlib.font_manager as fm

def setup_japanese_font():
    """日本語フォントを確実に設定する"""
    # Step1: japanize_matplotlib を試みる（最も確実）
    try:
        import japanize_matplotlib  # noqa
        plt.rcParams['axes.unicode_minus'] = False
        return "japanize_matplotlib"
    except ImportError:
        pass

    # Step2: Noto Sans CJK（Linux/Cloud環境に多い）
    cjk_fonts = [f.name for f in fm.fontManager.ttflist
                 if any(k in f.name for k in ['Noto', 'CJK', 'Gothic', 'Mincho', 'Meiryo', 'Yu Gothic'])]
    if cjk_fonts:
        plt.rcParams['font.family'] = cjk_fonts[0]
        plt.rcParams['axes.unicode_minus'] = False
        return cjk_fonts[0]

    # Step3: フォントファイルを直接検索
    import os, glob
    search_dirs = ['/usr/share/fonts', '/usr/local/share/fonts',
                   os.path.expanduser('~/.fonts'), 'C:/Windows/Fonts']
    for d in search_dirs:
        for ext in ['*.ttf', '*.otf']:
            for path in glob.glob(os.path.join(d, '**', ext), recursive=True):
                name = os.path.basename(path).lower()
                if any(k in name for k in ['gothic', 'mincho', 'noto', 'cjk', 'meiryo']):
                    prop = fm.FontProperties(fname=path)
                    matplotlib.font_manager.fontManager.addfont(path)
                    plt.rcParams['font.family'] = prop.get_name()
                    plt.rcParams['axes.unicode_minus'] = False
                    return path

    # Step4: ASCII代替（最終手段）
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    return "fallback"

_font_result = setup_japanese_font()

def get_font_prop():
    """matplotlib用FontPropertiesを返す（文字化け対策）"""
    try:
        import japanize_matplotlib  # noqa
        return None  # japanize_matplotlib が設定済みなのでNoneでOK
    except ImportError:
        pass
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ['Noto', 'CJK', 'Gothic', 'Mincho', 'Meiryo']):
            return fm.FontProperties(family=f.name)
    return None

# ページ設定
st.set_page_config(
    page_title="NPB選手年俸予測システム",
    page_icon="⚾",
    layout="centered",
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    position: fixed !important; top: 0; left: 0;
    width: 280px !important; height: 100vh !important;
    background-color: #ffe4e9 !important;
    border-right: 1px solid #e0e0e0;
    z-index: 1000000;
    border-radius: 0px 30px 30px 0;
    overflow: hidden;
}
[data-testid="stSidebarContent"] {
    overflow-y: auto !important; height: 100vh !important;
    padding: 0 0.5rem 1rem 0.5rem !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    font-size: 13px !important; line-height: 1.2 !important;
}
.main { margin-left: 280px !important; }
.block-container { max-width: 1400px !important; padding-top: 2rem !important; }
[data-testid="stHeaderActionElements"] { display: none !important; }
* { animation-duration: 0s !important; transition-duration: 0s !important; }
@media (max-width: 900px) {
    [data-testid="stSidebar"] { position: relative !important; width: 100% !important; height: auto !important; }
    .main { margin-left: 0 !important; }
}
</style>
""", unsafe_allow_html=True)

# タイトル
st.title("⚾ NPB選手年俸予測システム")

# フォント警告
if _font_result == "fallback":
    st.warning("⚠️ 日本語フォントが見つかりません。requirements.txt に `japanize-matplotlib` を追加してください。")

st.markdown("---")

# ============================================================
# セッション状態の初期化
# ============================================================
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ============================================================
# 減額制限計算関数
# ============================================================
def calculate_salary_limit(previous_salary):
    if previous_salary >= 100_000_000:
        return previous_salary * 0.60, 0.40
    else:
        return previous_salary * 0.75, 0.25

def check_salary_reduction_limit(predicted_salary, previous_salary):
    min_salary, reduction_rate = calculate_salary_limit(previous_salary)
    if predicted_salary < min_salary:
        return True, min_salary, reduction_rate
    return False, min_salary, reduction_rate

# ============================================================
# 予測履歴追加
# ============================================================
def add_to_history(player_name, predict_year, predicted_salary, actual_salary,
                   previous_salary, stats_dict, model_name,
                   is_limited=False, limited_salary=None):
    jst_time = datetime.utcnow() + timedelta(hours=9)
    item = {
        '予測日時': jst_time.strftime('%Y-%m-%d %H:%M:%S'),
        '選手名': player_name, '予測年度': predict_year,
        '予測年俸': predicted_salary,
        '制限後年俸': limited_salary if is_limited else predicted_salary,
        '実際の年俸': actual_salary, '前年年俸': previous_salary,
        '減額制限': is_limited, 'モデル': model_name, '成績': stats_dict
    }
    st.session_state.prediction_history.insert(0, item)
    if len(st.session_state.prediction_history) > 20:
        st.session_state.prediction_history = st.session_state.prediction_history[:20]

# ============================================================
# グラフ描画ヘルパー
# ============================================================
def set_labels(ax, xlabel='', ylabel='', title=''):
    """日本語ラベルをフォント設定込みで付与"""
    fp = get_font_prop()
    if fp:
        if xlabel: ax.set_xlabel(xlabel, fontproperties=fp, fontweight='bold')
        if ylabel: ax.set_ylabel(ylabel, fontproperties=fp, fontweight='bold')
        if title:  ax.set_title(title,  fontproperties=fp, fontweight='bold')
        # 軸ティックラベル
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(fp)
    else:
        if xlabel: ax.set_xlabel(xlabel, fontweight='bold')
        if ylabel: ax.set_ylabel(ylabel, fontweight='bold')
        if title:  ax.set_title(title,  fontweight='bold')

def legend_jp(ax, **kwargs):
    """凡例に日本語フォントを適用"""
    fp = get_font_prop()
    leg = ax.legend(**kwargs)
    if fp and leg:
        for text in leg.get_texts():
            text.set_fontproperties(fp)
    return leg

# ============================================================
# データ読み込み
# ============================================================
@st.cache_data
def load_data():
    try:
        salary_df  = pd.read_csv('data/salary_2023&2024&2025.csv')
        s2023      = pd.read_csv('data/stats_2023.csv')
        s2024      = pd.read_csv('data/stats_2024.csv')
        s2025      = pd.read_csv('data/stats_2025.csv')
        titles_df  = pd.read_csv('data/titles_2023&2024&2025.csv')
        return salary_df, s2023, s2024, s2025, titles_df, True
    except FileNotFoundError:
        return None, None, None, None, None, False

salary_df, stats_2023, stats_2024, stats_2025, titles_df, data_loaded = load_data()

if not data_loaded:
    st.sidebar.markdown("**5つのCSVを一度に選択してアップロード：**")
    uploaded_files = st.sidebar.file_uploader(
        "CSVファイル（5つ全て選択）", type=['csv'], accept_multiple_files=True
    )
    if uploaded_files and len(uploaded_files) == 5:
        file_dict = {}
        for f in uploaded_files:
            n = f.name.lower()
            if 'salary' in n or '年俸' in n:    file_dict['salary']     = f
            elif 'title' in n or 'タイトル' in n: file_dict['titles']  = f
            elif '2023' in n: file_dict['stats_2023'] = f
            elif '2024' in n: file_dict['stats_2024'] = f
            elif '2025' in n: file_dict['stats_2025'] = f
        if len(file_dict) == 5:
            salary_df  = pd.read_csv(file_dict['salary'])
            stats_2023 = pd.read_csv(file_dict['stats_2023'])
            stats_2024 = pd.read_csv(file_dict['stats_2024'])
            stats_2025 = pd.read_csv(file_dict['stats_2025'])
            titles_df  = pd.read_csv(file_dict['titles'])
            data_loaded = True
        else:
            st.sidebar.error("❌ ファイル名が正しくありません")
    elif uploaded_files:
        st.sidebar.warning(f"⚠️ {len(uploaded_files)}個選択中（5つ必要）")

# ============================================================
# データ前処理
# ============================================================
@st.cache_data
def prepare_data(_salary_df, _s23, _s24, _s25, _titles_df):
    titles_clean   = _titles_df.dropna(subset=['選手名'])
    title_summary  = titles_clean.groupby(['選手名', '年度']).size().reset_index(name='タイトル数')

    s23, s24, s25 = _s23.copy(), _s24.copy(), _s25.copy()
    s23['年度'], s24['年度'], s25['年度'] = 2023, 2024, 2025
    stats_all = pd.concat([s23, s24, s25], ignore_index=True)

    def make_salary(df, col_name, col_year, year):
        d = df[['選手名', col_year]].copy()
        d['年度'] = year
        d.rename(columns={col_year: '年俸_円'}, inplace=True)
        return d

    salary_long = pd.concat([
        make_salary(_salary_df, '選手名', '年俸_円_2023', 2023),
        make_salary(_salary_df, '選手名', '年俸_円_2024', 2024),
        make_salary(_salary_df, '選手名', '年俸_円_2025', 2025),
    ], ignore_index=True)
    salary_long = salary_long.dropna(subset=['年俸_円'])
    salary_long = salary_long[salary_long['年俸_円'] > 0]
    salary_long = salary_long.sort_values('年俸_円', ascending=False)
    salary_long = salary_long.drop_duplicates(subset=['選手名', '年度'], keep='first')

    stats_all['予測年度'] = stats_all['年度'] + 1
    merged = pd.merge(stats_all, title_summary, on=['選手名', '年度'], how='left')
    merged['タイトル数'] = merged['タイトル数'].fillna(0)
    merged = pd.merge(merged, salary_long,
                      left_on=['選手名', '予測年度'],
                      right_on=['選手名', '年度'],
                      suffixes=('_成績', '_年俸'))
    merged = merged.drop(columns=['年度_年俸', '予測年度'])
    merged.rename(columns={'年度_成績': '成績年度'}, inplace=True)

    stats_with_titles = pd.merge(stats_all, title_summary, on=['選手名', '年度'], how='left')
    stats_with_titles['タイトル数'] = stats_with_titles['タイトル数'].fillna(0)

    return merged, stats_with_titles, salary_long

# ============================================================
# モデル訓練
# ============================================================
@st.cache_resource
def train_models(_merged_df):
    base_features = ['試合', '打席', '打数', '得点', '安打', '二塁打', '三塁打', '本塁打',
                     '塁打', '打点', '盗塁', '盗塁刺', '四球', '死球', '三振', '併殺打',
                     '打率', '出塁率', '長打率', '犠打', '犠飛', 'タイトル数']
    feature_cols = base_features + (['年齢'] if '年齢' in _merged_df.columns else [])

    ml_df = _merged_df[feature_cols + ['年俸_円', '選手名', '成績年度']].copy()
    if '年齢' not in ml_df.columns:
        ml_df['年齢'] = 28
        feature_cols = base_features + ['年齢']
    ml_df = ml_df.dropna()

    X, y = ml_df[feature_cols], ml_df['年俸_円']
    y_log = np.log1p(y)
    X_train, X_test, ytr_log, yte_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    yte_orig = np.expm1(yte_log)

    scaler = StandardScaler()
    Xtr_sc, Xte_sc = scaler.fit_transform(X_train), scaler.transform(X_test)

    model_defs = {
        '線形回帰':         LinearRegression(),
        'ランダムフォレスト': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        '勾配ブースティング': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    results = {}
    for name, mdl in model_defs.items():
        if name == '線形回帰':
            mdl.fit(Xtr_sc, ytr_log); pred_log = mdl.predict(Xte_sc)
        else:
            mdl.fit(X_train, ytr_log); pred_log = mdl.predict(X_test)
        pred = np.expm1(pred_log)
        results[name] = {'model': mdl,
                         'MAE': mean_absolute_error(yte_orig, pred),
                         'R2':  r2_score(yte_orig, pred)}

    best_name  = max(results, key=lambda k: results[k]['R2'])
    best_model = results[best_name]['model']
    return best_model, best_name, scaler, feature_cols, results, ml_df

# ============================================================
# 予測ヘルパー
# ============================================================
def predict_salary(player_stats_row, feature_cols):
    fc = feature_cols
    if '年齢' in fc and '年齢' not in player_stats_row.index:
        vals = player_stats_row[fc[:-1]].values.tolist() + [28]
        features = np.array([vals])
    else:
        features = player_stats_row[fc].values.reshape(1, -1)

    if st.session_state.best_model_name == '線形回帰':
        pred_log = st.session_state.best_model.predict(
            st.session_state.scaler.transform(features))[0]
    else:
        pred_log = st.session_state.best_model.predict(features)[0]

    salary = np.expm1(pred_log)
    return round(salary / 100000) * 100000

# ============================================================
# データ読み込み & モデル訓練
# ============================================================
if data_loaded:
    if not st.session_state.model_trained:
        with st.spinner('🤖 モデルを訓練中...'):
            merged_df, stats_all_with_titles, salary_long = prepare_data(
                salary_df, stats_2023, stats_2024, stats_2025, titles_df)
            best_model, best_model_name, scaler, feature_cols, results, ml_df = \
                train_models(merged_df)
            st.session_state.update({
                'model_trained':         True,
                'best_model':            best_model,
                'best_model_name':       best_model_name,
                'scaler':                scaler,
                'feature_cols':          feature_cols,
                'stats_all_with_titles': stats_all_with_titles,
                'salary_long':           salary_long,
                'results':               results,
                'ml_df':                 ml_df,
            })

    # ============================================================
    # メニュー
    # ============================================================
    st.sidebar.markdown("### 🎯 機能選択")
    menu = st.sidebar.radio("", [
        "🏠 ホーム", "🔍 選手予測", "📊 選手比較",
        "🔬 モデル比較", "✏️ カスタム", "📈 性能",
        "📉 要因分析", "🏆 精度ランキング", "💰 年俸別予測", "📜 予測履歴"
    ], label_visibility="collapsed")

    # ============================================================
    # 🏠 ホーム
    # ============================================================
    if menu == "🏠 ホーム":
        c1, c2 = st.columns(2)
        c1.metric("採用モデル", st.session_state.best_model_name)
        c2.metric("R²スコア", f"{st.session_state.results[st.session_state.best_model_name]['R2']:.4f}")

        st.subheader("📖 使い方")
        st.markdown("""
        1. 左サイドバーからメニューを選択
        2. 選手名を入力して年俸を予測

        ### 機能一覧
        - 🔍 **選手予測**: 個別選手の年俸予測・レーダーチャート
        - 📊 **選手比較**: 最大5人の選手を比較
        - 🔬 **モデル比較**: 全モデルで同時予測
        - ✏️ **カスタム**: 自分でデータを入力して予測
        - 📈 **性能**: モデルの詳細評価
        - 📉 **要因分析**: 年俸に影響する要因を分析
        - 🏆 **精度ランキング**: 誤差の少ない選手を分析
        - 💰 **年俸別予測**: 年俸レンジ別の特化モデル
        - 📜 **予測履歴**: 過去20件の予測履歴

        ### ⚖️ NPB減額制限
        - **1億円以上**: 最大40%減額（最低60%保証）
        - **1億円未満**: 最大25%減額（最低75%保証）
        """)

    # ============================================================
    # 🔍 選手予測
    # ============================================================
    elif menu == "🔍 選手予測":
        st.header("🔍 選手検索・予測")

        avail = sorted(st.session_state.stats_all_with_titles[
            st.session_state.stats_all_with_titles['年度'] == 2024
        ]['選手名'].unique())

        kw = st.text_input("🔍 絞り込み検索", placeholder="例: 村上、岡本")
        filtered = [p for p in avail if kw in p] if kw else avail
        if kw and not filtered:
            st.warning("⚠️ 該当なし"); filtered = avail

        sel = st.selectbox("選手を選択", filtered)
        yr  = st.slider("予測年度", 2024, 2026, 2025)

        if st.button("🎯 予測実行", type="primary"):
            sy = yr - 1
            row = st.session_state.stats_all_with_titles[
                (st.session_state.stats_all_with_titles['選手名'] == sel) &
                (st.session_state.stats_all_with_titles['年度'] == sy)
            ]
            if row.empty:
                st.error(f"❌ {sel} の {sy} 年データが見つかりません")
            else:
                row = row.iloc[0]
                pred = predict_salary(row, st.session_state.feature_cols)

                prev_data = st.session_state.salary_long[
                    (st.session_state.salary_long['選手名'] == sel) &
                    (st.session_state.salary_long['年度'] == sy)]
                prev = prev_data['年俸_円'].values[0] if not prev_data.empty else None

                act_data = st.session_state.salary_long[
                    (st.session_state.salary_long['選手名'] == sel) &
                    (st.session_state.salary_long['年度'] == yr)]
                act = act_data['年俸_円'].values[0] if not act_data.empty else None

                disp, is_lim = pred, False
                if prev:
                    is_lim, min_sal, rate = check_salary_reduction_limit(pred, prev)
                    if is_lim:
                        disp = min_sal
                        st.warning(f"⚖️ 減額制限: 前年{prev/10000:.0f}万円 → 制限後最低{min_sal/10000:.0f}万円（{rate*100:.0f}%制限）")

                st.success("✅ 予測完了！")

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("前年年俸",  f"{prev/10000:.0f}万円" if prev else "なし")
                c2.metric("予測年俸",  f"{pred/10000:.0f}万円")
                c3.metric("実際の年俸", f"{act/10000:.0f}万円" if act else "なし")
                if act:
                    c4.metric("予測誤差", f"{abs(disp-act)/act*100:.1f}%")

                st.markdown("---")
                st.subheader(f"{sy}年の成績")
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("試合", int(row['試合']));    c1.metric("打率",  f"{row['打率']:.3f}")
                c2.metric("安打", int(row['安打']));    c2.metric("出塁率", f"{row['出塁率']:.3f}")
                c3.metric("本塁打", int(row['本塁打'])); c3.metric("長打率", f"{row['長打率']:.3f}")
                c4.metric("打点", int(row['打点']));    c4.metric("タイトル数", int(row['タイトル数']))
                if '年齢' in row.index: c5.metric("年齢", int(row['年齢']))

                # 履歴に保存
                add_to_history(sel, yr, pred, act, prev,
                    {'試合':int(row['試合']),'安打':int(row['安打']),
                     '本塁打':int(row['本塁打']),'打点':int(row['打点']),
                     '打率':float(row['打率']),'出塁率':float(row['出塁率']),
                     '長打率':float(row['長打率']),'タイトル数':int(row['タイトル数']),
                     '年齢':int(row['年齢']) if '年齢' in row.index else 28},
                    st.session_state.best_model_name, is_lim, disp if is_lim else None)

                st.markdown("---")
                c1, c2 = st.columns(2)

                # 年俸推移グラフ
                with c1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    hist = st.session_state.salary_long[
                        st.session_state.salary_long['選手名'] == sel
                    ].sort_values('年度')
                    if not hist.empty:
                        ax.plot(hist['年度'].astype(int), hist['年俸_円']/10000,
                                'o-', lw=2, ms=8, label='実際の年俸')
                        ax.plot(yr, pred/10000, 'r*', ms=20, label='予測年俸')
                        if is_lim:
                            ax.plot(yr, disp/10000, 'D', ms=12, color='orange', label='制限後年俸')
                        if act:
                            ax.plot(yr, act/10000, 'go', ms=12, label=f'実際({yr})')
                    ax.set_xticks([2023, 2024, 2025, 2026])
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: int(x)))
                    set_labels(ax, '年度', '年俸（万円）', f'{sel} - 年俸推移')
                    legend_jp(ax)
                    ax.grid(alpha=0.3)
                    st.pyplot(fig); plt.close(fig)

                # レーダーチャート
                with c2:
                    fig, ax = plt.subplots(figsize=(8,5), subplot_kw=dict(projection='polar'))
                    radar = {
                        '打率':  row['打率']/0.4,
                        '出塁率': row['出塁率']/0.5,
                        '長打率': row['長打率']/0.7,
                        '本塁打': min(row['本塁打']/40, 1.0),
                        '打点':  min(row['打点']/100, 1.0),
                        '盗塁':  min(row['盗塁']/40, 1.0),
                    }
                    cats = list(radar.keys())
                    vals = list(radar.values()) + [list(radar.values())[0]]
                    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
                    angles += angles[:1]
                    ax.plot(angles, vals, 'o-', lw=2, color='#2E86AB')
                    ax.fill(angles, vals, alpha=0.25, color='#2E86AB')
                    ax.set_xticks(angles[:-1])
                    fp = get_font_prop()
                    ax.set_xticklabels(cats, fontproperties=fp) if fp else ax.set_xticklabels(cats)
                    ax.set_ylim(0,1)
                    title_txt = f'{sel} - 成績レーダー\n({sy}年)'
                    ax.set_title(title_txt, fontproperties=fp, fontweight='bold', pad=20) if fp \
                        else ax.set_title(title_txt, fontweight='bold', pad=20)
                    ax.grid(True)
                    st.pyplot(fig); plt.close(fig)

    # ============================================================
    # 📊 選手比較
    # ============================================================
    elif menu == "📊 選手比較":
        st.header("📊 複数選手比較")
        avail = sorted(st.session_state.stats_all_with_titles[
            st.session_state.stats_all_with_titles['年度'] == 2024
        ]['選手名'].unique())
        sels = st.multiselect("比較する選手を2人以上選択（最大5人）", avail, max_selections=5)

        if len(sels) >= 2 and st.button("📊 比較実行", type="primary"):
            rows_list = []
            for p in sels:
                r = st.session_state.stats_all_with_titles[
                    (st.session_state.stats_all_with_titles['選手名'] == p) &
                    (st.session_state.stats_all_with_titles['年度'] == 2024)
                ]
                if r.empty: continue
                r = r.iloc[0]
                pred = predict_salary(r, st.session_state.feature_cols)
                prev_d = st.session_state.salary_long[
                    (st.session_state.salary_long['選手名'] == p) &
                    (st.session_state.salary_long['年度'] == 2024)]
                prev = prev_d['年俸_円'].values[0] if not prev_d.empty else None
                disp, is_lim = pred, False
                if prev:
                    is_lim, ms, _ = check_salary_reduction_limit(pred, prev)
                    if is_lim: disp = ms
                rows_list.append({
                    '選手名': p,
                    '前年年俸(万円)':    prev/10000 if prev else None,
                    '予測年俸(制限前万円)': pred/10000,
                    '予測年俸(制限後万円)': disp/10000,
                    '減額制限': 'あり' if is_lim else 'なし',
                    '打率': r['打率'], '本塁打': int(r['本塁打']),
                    '打点': int(r['打点']), 'タイトル数': int(r['タイトル数'])
                })

            if rows_list:
                df = pd.DataFrame(rows_list)
                st.dataframe(df, use_container_width=True, hide_index=True)

                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(8,5))
                    x = np.arange(len(df)); w = 0.35
                    ax.barh(x-w/2, df['予測年俸(制限前万円)'], w, label='予測（制限前）', alpha=0.7, color='steelblue')
                    ax.barh(x+w/2, df['予測年俸(制限後万円)'], w, label='予測（制限後）', alpha=0.7, color='orange')
                    ax.set_yticks(x)
                    fp = get_font_prop()
                    ax.set_yticklabels(df['選手名'], fontproperties=fp) if fp else ax.set_yticklabels(df['選手名'])
                    set_labels(ax, '予測年俸（万円）', '', '予測年俸比較')
                    legend_jp(ax); ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig); plt.close(fig)
                with c2:
                    fig, ax = plt.subplots(figsize=(8,5))
                    x = np.arange(len(df)); w = 0.25
                    ax.bar(x-w, df['打率']*100, w, label='打率 x100', alpha=0.8)
                    ax.bar(x,   df['本塁打'],   w, label='本塁打',     alpha=0.8)
                    ax.bar(x+w, df['打点']/10,  w, label='打点 /10',   alpha=0.8)
                    ax.set_xticks(x)
                    ax.set_xticklabels(df['選手名'], rotation=45, ha='right',
                                       fontproperties=fp) if fp else \
                    ax.set_xticklabels(df['選手名'], rotation=45, ha='right')
                    set_labels(ax, '選手', '値（正規化）', '成績比較')
                    legend_jp(ax); ax.grid(axis='y', alpha=0.3)
                    st.pyplot(fig); plt.close(fig)

    # ============================================================
    # 📈 性能
    # ============================================================
    elif menu == "📈 性能":
        st.header("📈 モデル性能")
        rows = [{'モデル': n, 'MAE（万円）': f"{r['MAE']/10000:.2f}", 'R²スコア': f"{r['R2']:.4f}"}
                for n, r in st.session_state.results.items()]
        st.dataframe(pd.DataFrame(rows).sort_values('R²スコア', ascending=False),
                     use_container_width=False, hide_index=True)
        st.success(f"🏆 最良モデル: {st.session_state.best_model_name}")

        if st.session_state.best_model_name == 'ランダムフォレスト':
            st.markdown("---")
            st.subheader("特徴量重要度 Top 10")
            fi = pd.DataFrame({
                '特徴量': st.session_state.feature_cols,
                '重要度': st.session_state.best_model.feature_importances_
            }).sort_values('重要度', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10,6))
            ax.barh(range(len(fi)), fi['重要度'], color='#9b59b6', alpha=0.7)
            ax.set_yticks(range(len(fi)))
            fp = get_font_prop()
            ax.set_yticklabels(fi['特徴量'], fontproperties=fp) if fp else \
            ax.set_yticklabels(fi['特徴量'])
            set_labels(ax, '重要度', '', '特徴量重要度 Top 10')
            ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig); plt.close(fig)

    # ============================================================
    # 📉 要因分析
    # ============================================================
    elif menu == "📉 要因分析":
        st.header("📉 要因分析")
        ml = st.session_state.ml_df

        tg = ml.groupby(ml['タイトル数'] > 0)['年俸_円'].agg(['count','mean','median'])
        tg['mean']   = round(tg['mean']   / 10000)
        tg['median'] = round(tg['median'] / 10000)
        tg.index   = ['タイトル無し', 'タイトル有り']
        tg.columns = ['選手数', '平均年俸（万円）', '中央値（万円）']
        st.subheader("タイトル獲得の影響")
        st.dataframe(tg, use_container_width=False)

        diff = tg.loc['タイトル有り','平均年俸（万円）'] - tg.loc['タイトル無し','平均年俸（万円）']
        st.metric("タイトル獲得による年俸増加", f"{round(diff)}万円")

        st.markdown("---")
        st.subheader("主要指標との相関")
        corr_cols = ['打率','本塁打','打点','出塁率','長打率','タイトル数','年齢','試合','年俸_円'] \
                    if '年齢' in ml.columns else \
                    ['打率','本塁打','打点','出塁率','長打率','タイトル数','試合','年俸_円']
        corr = ml[corr_cols].corr()['年俸_円'].sort_values(ascending=False)
        st.dataframe(pd.DataFrame([{'指標': i, '相関係数': f"{v:.4f}"}
                                   for i,v in corr.items() if i != '年俸_円']),
                     use_container_width=False, hide_index=True)

        for col, color, title in [
            ('打率',  'steelblue', '打率と年俸の関係'),
            ('本塁打', 'orange',   '本塁打と年俸の関係'),
        ] + ([('年齢', 'green', '年齢と年俸の関係')] if '年齢' in ml.columns else []):
            fig, ax = plt.subplots(figsize=(8,5))
            ax.scatter(ml[col], ml['年俸_円']/10000, alpha=0.5, color=color)
            set_labels(ax, col, '年俸（万円）', title)
            ax.grid(alpha=0.3)
            st.pyplot(fig); plt.close(fig)

    # ============================================================
    # 📜 予測履歴
    # ============================================================
    elif menu == "📜 予測履歴":
        st.header("📜 予測履歴")
        if not st.session_state.prediction_history:
            st.info("📭 履歴がありません。選手予測を実行すると保存されます。")
        else:
            st.markdown(f"**保存件数**: {len(st.session_state.prediction_history)} / 20件")
            if st.button("🗑️ 履歴をクリア", type="secondary"):
                st.session_state.prediction_history = []
                st.rerun()
            st.markdown("---")
            for idx, item in enumerate(st.session_state.prediction_history):
                with st.expander(
                    f"#{idx+1} {item['選手名']} - {item['予測年度']}年 ({item['予測日時']})",
                    expanded=(idx == 0)
                ):
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("前年年俸", f"{item['前年年俸']/10000:.1f}万円" if item['前年年俸'] else "なし")
                    c2.metric("予測年俸", f"{item['予測年俸']/10000:.1f}万円")
                    c3.metric("制限後年俸", f"{item['制限後年俸']/10000:.1f}万円" if item['減額制限'] else "制限なし")
                    if item['実際の年俸']:
                        c4.metric("実際の年俸", f"{item['実際の年俸']/10000:.1f}万円")
                        err = abs(item['制限後年俸'] - item['実際の年俸']) / item['実際の年俸'] * 100
                        st.metric("誤差率", f"{err:.1f}%")
                    if item['減額制限']: st.warning("⚖️ 減額制限が適用されました")

                    st.markdown(f"**使用モデル**: {item['モデル']}")
                    s = item['成績']
                    c1,c2,c3,c4,c5 = st.columns(5)
                    c1.metric("試合", s['試合']); c2.metric("安打", s['安打'])
                    c3.metric("本塁打", s['本塁打']); c4.metric("打点", s['打点'])
                    c5.metric("年齢", f"{s['年齢']}歳")

            # CSVエクスポート
            st.markdown("---")
            export = [{'予測日時':i['予測日時'],'選手名':i['選手名'],
                       '予測年俸(万円)': round(i['予測年俸']/10000,1),
                       '実際(万円)': round(i['実際の年俸']/10000,1) if i['実際の年俸'] else None,
                       '減額制限': '有' if i['減額制限'] else '無',
                       'モデル': i['モデル']}
                      for i in st.session_state.prediction_history]
            csv = pd.DataFrame(export).to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 CSVでダウンロード", csv,
                               f"history_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                               mime="text/csv")

    else:
        st.info("🚧 このメニューは別ファイル（npb_salary_app_full.py）に実装されています")

else:
    st.info("👈 左サイドバーから5つのCSVをアップロードしてください")

st.markdown("---")
st.markdown("*NPB選手年俸予測システム - Powered by Streamlit*")
