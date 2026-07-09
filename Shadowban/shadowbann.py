# =============================================================================
# 1. IMPORTS UND KONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import MNLogit
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path


BASE_DIR = Path(__file__).parent
# =============================================================================
# 2. KONFIGURATION & PFADE
# =============================================================================
CSV_VIDEOS = BASE_DIR / "data" / "sample.csv"
CACHE_FILE = BASE_DIR / "data" / "cache.csv"



# Parameter für Upload-Frequenz-Analyse
C_MAX_UPLOADS_PER_MONTH = 60
BIN_QS = [0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]
BIN_LABELS = ["Q1", "Q2", "Q3", "Q4", "P90", "Top5%"]
TICK_VALUES = [1, 3, 10, 30, 100, 300, 500]
Y_MIN, Y_MAX = 0.0, 3.2

# =============================================================================
# 3. HILFSFUNKTIONEN
# =============================================================================

def to_numeric(s: pd.Series) -> pd.Series:
    """Wandelt kommagetrennte Zahlenstrings in floats um."""
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )

def build_channel_freq(df_videos: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Video-Daten auf Kanalebene:
    - Gesamtuploads, erstes/letztes Datum, aktive Tage, Uploads pro Monat,
      mittlerer SearchFound-Wert.
    """
    channel_freq = (
        df_videos.groupby("channel_id")
        .agg(
            uploads_total=("video_id", "count"),
            first_upload=("published_at", "min"),
            last_upload=("published_at", "max"),
            mean_searchfound=("SearchFound_scaled", "mean"),
        )
        .reset_index()
    )
    channel_freq["active_days"] = (
        (channel_freq["last_upload"] - channel_freq["first_upload"])
        .dt.days
        .clip(lower=1)
    )
    channel_freq["uploads_per_month"] = (
        channel_freq["uploads_total"] / channel_freq["active_days"] * 30
    )
    return channel_freq

def plot_quadratic_log(ax, reg_df: pd.DataFrame, x_col: str, title: str, upper_limit: float):
    """Scatterplot + quadratische log-Regressionskurve mit 95%-KI (geclippt auf 0..3)."""
    ax.scatter(reg_df[x_col], reg_df["mean_searchfound"], alpha=0.35, s=18, edgecolor="none")

    tmp = reg_df.copy()
    tmp["log_x"] = np.log1p(tmp[x_col])
    tmp["log_x_c"] = tmp["log_x"] - tmp["log_x"].mean()
    tmp["log_x_sq"] = tmp["log_x_c"] ** 2

    X = sm.add_constant(tmp[["log_x_c", "log_x_sq"]])
    y = tmp["mean_searchfound"]
    model = sm.OLS(y, X).fit(cov_type="HC3")

    x_pred = np.linspace(0.1, upper_limit, 300)
    x_log = np.log1p(x_pred)
    x_c = x_log - tmp["log_x"].mean()
    X_pred = pd.DataFrame({
        "const": 1.0,
        "log_x_c": x_c,
        "log_x_sq": x_c ** 2
    })[model.params.index]
    pred = model.get_prediction(X_pred).summary_frame(alpha=0.05)
    pred["mean"] = pred["mean"].clip(0, 3)
    pred["mean_ci_lower"] = pred["mean_ci_lower"].clip(0, 3)
    pred["mean_ci_upper"] = pred["mean_ci_upper"].clip(0, 3)

    ax.plot(x_pred, pred["mean"], linewidth=2, label="Regressionslinie")
    ax.fill_between(x_pred, pred["mean_ci_lower"], pred["mean_ci_upper"], alpha=0.2, label="95%-KI")

    ax.set_xscale("log")
    ticks = [t for t in TICK_VALUES if t <= upper_limit]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(1, upper_limit)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(alpha=0.25, which="both", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Uploads pro Monat (logarithmisch skaliert)")
    ax.set_ylabel("Ø Sichtbarkeit (SearchFound 0–3)")
    return model

def plot_binned(ax, reg_df: pd.DataFrame, x_col: str, title: str):
    """Option B: Quantil-Bins mit Dummy-Regression."""
    dfb = reg_df.dropna(subset=[x_col, "mean_searchfound"]).copy()
    dfb["upload_bin"] = pd.qcut(dfb[x_col], q=BIN_QS, labels=BIN_LABELS, duplicates="drop")
    dfb = dfb.dropna(subset=["upload_bin"])
    dummies = pd.get_dummies(dfb["upload_bin"], prefix="bin", drop_first=True).astype(float)
    X = sm.add_constant(dummies).astype(float)
    y = dfb["mean_searchfound"].astype(float)
    model = sm.OLS(y, X).fit(cov_type="HC3")

    bin_summary = dfb.groupby("upload_bin", observed=True).agg(
        x_med=(x_col, "median"), n=("mean_searchfound", "size")
    ).reset_index()

    X_pred_rows = []
    for bin_label in bin_summary["upload_bin"]:
        row = {col: 0.0 for col in X.columns}
        row["const"] = 1.0
        dummy_col = f"bin_{bin_label}"
        if dummy_col in row:
            row[dummy_col] = 1.0
        X_pred_rows.append(row)
    X_pred = pd.DataFrame(X_pred_rows)[X.columns].astype(float)
    pred = model.get_prediction(X_pred).summary_frame(alpha=0.05)
    pred["mean"] = pred["mean"].clip(0, 3)
    pred["mean_ci_lower"] = pred["mean_ci_lower"].clip(0, 3)
    pred["mean_ci_upper"] = pred["mean_ci_upper"].clip(0, 3)

    ax.scatter(dfb[x_col], dfb["mean_searchfound"], alpha=0.08, s=10, edgecolor="none")
    x_vals = bin_summary["x_med"].to_numpy()
    ax.plot(x_vals, pred["mean"], linewidth=2, label="Bin-Mittel (Dummy-Modell)")
    ax.fill_between(x_vals, pred["mean_ci_lower"], pred["mean_ci_upper"], alpha=0.2, label="95%-KI")

    ax.set_xscale("log")
    upper_limit = float(np.nanmax(dfb[x_col]))
    ticks = [t for t in TICK_VALUES if t <= upper_limit]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(1, upper_limit)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(alpha=0.25, which="both", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Uploads pro Monat (logarithmisch skaliert)")
    ax.set_ylabel("Ø Sichtbarkeit (SearchFound 0–3)")
    for i, lbl in enumerate(bin_summary["upload_bin"].astype(str)):
        ax.annotate(lbl, (x_vals[i], pred["mean"].iloc[i]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    return model

def load_topic_cache():
    """Lädt vorab abgerufene YouTube-Themenkategorien aus CSV-Cache."""
    if os.path.exists(CACHE_FILE):
        topics_df = pd.read_csv(CACHE_FILE)
        topics_df["video_id"] = topics_df["video_id"].astype(str)
        topics_df = topics_df.drop_duplicates(subset="video_id", keep="last")
        print(" Themen-Cache geladen.")
        return topics_df
    else:
        print("Keine Themen-Cache-Datei gefunden. Themen-Spalten bleiben leer.")
        return pd.DataFrame(columns=["video_id", "topicCategories", "n_topics"])

# =============================================================================
# 4. DATEN LADEN & GRUNDLEGENDE VORVERARBEITUNG
# =============================================================================

print(" Lade Videodaten...")
usecols = [
    "channel_id", "video_id", "published_at", "SearchFound_scaled",
    "like_count", "comment_count", "partisan", "view_count", "SubscriberCount",
    "title", "description", "live_broadcast_content", "tags", "duration_seconds",
    "duration", "description_word_count", "category_id"
]
df_raw = pd.read_csv(CSV_VIDEOS, usecols=usecols, low_memory=False)


sample_df = df_raw.groupby("channel_id").apply(
    lambda x: x.sample(frac=0.01, random_state=42)
).reset_index(drop=True)
sample_df.to_csv("kfsj.csv")


df_raw["published_at"] = pd.to_datetime(df_raw["published_at"], errors="coerce")
df_raw = df_raw.dropna(subset=["channel_id", "video_id", "published_at", "SearchFound_scaled"])



# Themen-Daten einfügen
topics_df = load_topic_cache()

df = df_raw.merge(topics_df, on="video_id", how="left")
df["n_topics"] = df["n_topics"].fillna(0).astype(int)
df["topicCategories"] = df["topicCategories"].fillna("")

print(f"Geladen: {len(df)} Videos von {df['channel_id'].nunique()} Kanälen.")

# =============================================================================
# 5. FEATURE ENGINEERING (ALLGEMEIN)
# =============================================================================

# Kanalaktivitätsmetriken (für alle Videos)
df["uploads_total"] = df.groupby("channel_id")["video_id"].transform("count")
df["first_upload"] = df.groupby("channel_id")["published_at"].transform("min")
df["last_upload"] = df.groupby("channel_id")["published_at"].transform("max")
df["active_days"] = (df["last_upload"] - df["first_upload"]).dt.days.clip(lower=1)
df["uploads_per_month"] = df["uploads_total"] / df["active_days"] * 30

# Log-Transformationen für schiefe Variablen
df["log_view_count"] = np.log1p(df["view_count"])
df["log_SubscriberCount"] = np.log1p(df["SubscriberCount"])

# Videos mit 0 Views entfernen (sehr wenige)
df = df[df["view_count"] > 0]

# Engagement-Features: fehlende Likes/Kommentare mit Median füllen
median_like = df["like_count"].median()
median_comment = df["comment_count"].median()
df["like_count"] = df["like_count"].fillna(median_like)
df["comment_count"] = df["comment_count"].fillna(median_comment)

# Engagement-Raten
df["total_engagement"] = df["like_count"] + df["comment_count"]
df["engagement_rate"] = (df["total_engagement"] / df["view_count"]) * 100
df["like_rate"] = (df["like_count"] / df["view_count"]) * 100
df["comment_rate"] = (df["comment_count"] / df["view_count"]) * 100

# Extreme Engagement-Ausreißer entfernen (3 Videos >100% Engagement)
extreme_videos = ["oo7u8j3yzDo", "XoKiWWEQU10", "oxpOAUhG7lc"]
df = df[~df["video_id"].isin(extreme_videos)]

# Log-Engagementmetriken
df["log_engagement_rate"] = np.log1p(df["engagement_rate"])
df["log_like_rate"] = np.log1p(df["like_rate"])
df["log_comment_rate"] = np.log1p(df["comment_rate"])

# Standardisierte Versionen (z‑Scores)
scaler = StandardScaler()
df[["like_rate_z", "comment_rate_z"]] = scaler.fit_transform(df[["like_rate", "comment_rate"]])
df["engagement_rate_z"] = (df["engagement_rate"] - df["engagement_rate"].mean()) / df["engagement_rate"].std()

# Kanal-Durchschnittswerte und Abweichungen
df["channel_avg_engament_rate"] = df.groupby("channel_id")["engagement_rate"].transform("mean")
df["channel_avg_like_rate"] = df.groupby("channel_id")["like_rate"].transform("mean")
df["channel_avg_comment_rate"] = df.groupby("channel_id")["comment_rate"].transform("mean")

df["engagement_delta"] = df["engagement_rate"] - df["channel_avg_engament_rate"]
df["like_delta"] = df["like_rate"] - df["channel_avg_like_rate"]
df["comment_delta"] = df["comment_rate"] - df["channel_avg_comment_rate"]

# View-Residuals (innerhalb des Kanals)
df["channel_mean_views"] = df.groupby("channel_id")["view_count"].transform("mean")
df["view_residual"] = df["view_count"] - df["channel_mean_views"]
df["normalized_view_resiual_per_channel"] = df.groupby("channel_id")["view_residual"].transform(
    lambda x: (x - x.mean()) / x.std()
)
df["normalized_view_residual_global"] = scaler.fit_transform(df[["view_residual"]])
df["log_channel_mean_views"] = np.log1p(df["channel_mean_views"])

# Bereinigung der Videodauer
df = df.dropna(subset=["duration_seconds"])
# Ein nicht mehr verfügbares Video entfernen
df = df[df["video_id"] != "l7Phjs9aiRU"]
# Videos länger als 2 Stunden entfernen
df = df[df["duration_seconds"] <= 7200]
df["log_duration"] = np.log1p(df["duration_seconds"])

# Titel und Tags
df["title_length"] = df["title"].astype(str).apply(len)
df["tags_count"] = df["tags"].astype(str).apply(lambda x: len(x.split("|")))

# Kanalgrößenklassifikation
def classify_channel_size(subscribers):
    if subscribers >= 1_000_000:
        return "Mega"
    elif subscribers >= 100_000:
        return "Macro"
    elif subscribers >= 10_000:
        return "Micro"
    else:
        return "Nano"

df["channel_size"] = df["SubscriberCount"].apply(classify_channel_size)

# Wortanzahl in der Beschreibung
df["description_word_count"] = df["description"].astype(str).apply(lambda x: len(x.split()))
df["log_description_words"] = np.log1p(df["description_word_count"])

# Shadowban-Messungen
df["shadowban_score_binary"] = (df["SearchFound_scaled"] <= 1).astype(int)
df["shadowban_gradual"] = (3 - df["SearchFound_scaled"]) / 3
df["ghost_ban"] = -df["normalized_view_resiual_per_channel"]

# Fehlende Partisan-Werte entfernen
df = df.dropna(subset=["partisan"])

# Zentrierung zur Reduktion von Multikollinearität
df["shadowban_gradual_c"] = df["shadowban_gradual"] - df["shadowban_gradual"].mean()
df["log_SubscriberCount_c"] = df["log_SubscriberCount"] - df["log_SubscriberCount"].mean()
df["partisan_c"] = df["partisan"] - df["partisan"].mean()
df["partisan_c_sq"] = df["partisan_c"] ** 2

# Topic-Diversität (aus früherer Berechnung rekonstruiert)
# Wir verwenden bereits die Spalte n_topics aus dem Cache
channel_topic_diversity = df.groupby("channel_id")["n_topics"].mean().reset_index(name="avg_topic_diversity")
df = df.merge(channel_topic_diversity, on="channel_id", how="left")
df["avg_topic_diversity_c"] = df["avg_topic_diversity"] - df["avg_topic_diversity"].mean()
df["partisan_x_diversity"] = df["partisan_c"] * df["avg_topic_diversity"]

# Indikator für Mega-Kanäle
df["Mega"] = (df["channel_size"] == "Mega").astype(int)

# Interaktionsterme für spätere Modelle
df["partisan_x_subs"] = df["partisan_c"] * df["log_SubscriberCount_c"]
df["mega_partisan"] = df["Mega"] * df["partisan_c"]

# Finale Bereinigung: fehlende Werte in Modellvariablen entfernen
# Zunächst definieren wir die benötigten Spalten für die Modelle
model_vars = [
    "SearchFound_scaled", "partisan_c", "log_duration", "title_length",
    "engagement_rate_z", "log_SubscriberCount_c", "uploads_per_month",
    "log_channel_mean_views", "like_delta", "comment_delta", "engagement_delta",
    "avg_topic_diversity_c", "shadowban_score_binary", "ghost_ban",
    "tags_count", "log_engagement_rate", "Mega", "partisan_x_subs",
    "mega_partisan", "partisan_c_sq", "avg_topic_diversity", "partisan_x_diversity"
]
# Sicherstellen, dass alle Spalten existieren
existing_model_vars = [v for v in model_vars if v in df.columns]
df = df.dropna(subset=existing_model_vars)

print(f"Finale Anzahl einzigartiger Kanäle: {df['channel_id'].nunique()}")
print(f"Finale Anzahl Videos: {len(df)}")

# =============================================================================
# 6. DESKRIPTIVE STATISTIKEN
# =============================================================================

print("\n--- Deskriptive Statistiken ---")
stats_vars = [
    "shadowban_score_binary", "SearchFound_scaled", "partisan",
    "view_count", "SubscriberCount", "duration_seconds", "title_length",
    "engagement_rate", "uploads_per_month", "avg_topic_diversity"
]
summary_table = df[stats_vars].describe().T
print(summary_table[['mean', 'std', 'min', 'max']])

print("\nVerteilung der Kanalgrößen:")
print(df["channel_size"].value_counts(normalize=True))

print("\nVerteilung der Shadowban-Scores (%)")
print(df['SearchFound_scaled'].value_counts(normalize=True).sort_index() * 100)

# =============================================================================
# 7. ANALYSE: UPLOAD-FREQUENZ vs. SICHTBARKEIT
# =============================================================================

print("\n" + "="*80)
print("ANALYSE UPLOAD-FREQUENZ vs. SICHTBARKEIT")
print("="*80)

# Kanalebene aggregieren
channel_freq = build_channel_freq(df)
channel_freq = channel_freq[channel_freq["uploads_total"] > 1].copy()

# Vier Datensätze für verschiedene Capping-Optionen
ul_99 = channel_freq["uploads_per_month"].quantile(0.99)
cap_95 = channel_freq["uploads_per_month"].quantile(0.95)

baseline = channel_freq.copy()
baseline["uploads_pm_used"] = baseline["uploads_per_month"].clip(upper=ul_99)

optA = channel_freq.copy()
optA["uploads_pm_used"] = optA["uploads_per_month"].clip(upper=cap_95)

optB = channel_freq.copy()
optB["uploads_pm_used"] = optB["uploads_per_month"]

optC = channel_freq[channel_freq["uploads_per_month"] <= C_MAX_UPLOADS_PER_MONTH].copy()
ul_C = optC["uploads_per_month"].quantile(0.99)
optC["uploads_pm_used"] = optC["uploads_per_month"].clip(upper=ul_C)

# 2x2-Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
(ax1, ax2), (ax3, ax4) = axes

print("\nSchätze Regressionsmodelle für Upload-Frequenz...")
m1 = plot_quadratic_log(ax1, baseline, "uploads_pm_used",
                        "Baseline (winsor 99%, aktive Tage)", ul_99)
m2 = plot_quadratic_log(ax2, optA, "uploads_pm_used",
                        "Option A (Cap bei 95%, aktive Tage)", cap_95)
m3 = plot_binned(ax3, optB, "uploads_pm_used",
                 "Option B (Quantil-Bins, Dummy-Modell)")
m4 = plot_quadratic_log(ax4, optC, "uploads_pm_used",
                        f"Option C (Beschränkt auf ≤ {C_MAX_UPLOADS_PER_MONTH}/Monat)", ul_C)

for ax in [ax1, ax2, ax3, ax4]:
    ax.legend(loc="upper right")

plt.tight_layout()
plt.show()

# Ausgaben der Modelle
print("\n--- Modell 1 (Baseline) ---")
print(m1.summary())
print("\n--- Modell 2 (Option A: Cap 95%) ---")
print(m2.summary())
print("\n--- Modell 3 (Option B: Bins) ---")
print(m3.summary())
print(f"\n--- Modell 4 (Option C: Beschränkt ≤ {C_MAX_UPLOADS_PER_MONTH}) ---")
print(m4.summary())

# Cook's Distance (einflussreiche Kanäle)
influence = m1.get_influence()
cooks_d, _ = influence.cooks_distance
baseline["cooks_d"] = cooks_d
n = baseline.shape[0]
cook_threshold = 4 / n

print(f"\nCook's D-Schwellwert ≈ {cook_threshold:.6f}")
top_cooks = baseline.sort_values("cooks_d", ascending=False).head(15)[
    ["channel_id", "uploads_per_month", "mean_searchfound", "cooks_d"]
]
print("\nTop 15 einflussreiche Kanäle (Cook's D):")
print(top_cooks)

plt.figure(figsize=(7,4))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.axhline(cook_threshold, color="red", linestyle="--", label="4/n Schwellwert")
plt.yscale("log")
plt.xlabel("Kanal-Index")
plt.ylabel("Cook's Distanz (log-Skala)")
plt.legend()
plt.tight_layout()
plt.show()

# Robustes Modell ohne hoch einflussreiche Kanäle
robust_sample = baseline[baseline["cooks_d"] <= cook_threshold].copy()
fig, ax = plt.subplots()
m1_robust = plot_quadratic_log(
    ax, robust_sample, "uploads_pm_used",
    "Robustes Modell (ohne einflussreiche Ausreißer)", ul_99
)
plt.show()

# =============================================================================
# 8. SHADOWBAN-MODELLE (HYPOTHESEN)
# =============================================================================

print("\n" + "="*80)
print("SHADOWBAN-ANALYSE – HYPOTHESENTESTS")
print("="*80)

# -----------------------------------------------------------------------------  
# H1a: Extreme Parteilichkeit → höhere Shadowban-Wahrscheinlichkeit  
# -----------------------------------------------------------------------------  
print("\n" + "="*70)
print("H1a: Ordered Logit (SearchFound_scaled ~ partisan + Kontrollen)")
print("="*70)

X_h1a = df[["partisan_c", "log_duration", "title_length",
            "engagement_rate_z", "log_SubscriberCount_c", "uploads_per_month"]]
model_h1a = OrderedModel(df["SearchFound_scaled"], X_h1a, distr="logit")
res_h1a = model_h1a.fit(method="bfgs")
print(res_h1a.summary())
print(f"McFadden R2: {res_h1a.prsquared}")

# Interaktion: Partisan × Abonnentenzahl (zentriert)
X_h1a_int = df[["partisan_c", "partisan_x_subs", "log_duration",
                "engagement_rate_z", "log_channel_mean_views", "uploads_per_month"]]
model_h1a_int = OrderedModel(df["SearchFound_scaled"], X_h1a_int, distr="logit")
res_h1a_int = model_h1a_int.fit(method="bfgs")
print("\nMit Interaktion (partisan × subscriber count):")
print(res_h1a_int.summary())

# Hinzufügen von Kanalgröße (Mega) und quadratischem Term
X_h1a_size = df[["Mega", "partisan_c", "log_duration", "engagement_rate_z",
                 "log_channel_mean_views", "uploads_per_month", "partisan_c_sq"]]
model_h1a_size = OrderedModel(df["SearchFound_scaled"], X_h1a_size, distr="logit")
res_h1a_size = model_h1a_size.fit(method="bfgs")
print("\nMit Mega-Indikator und quadratischer Parteilichkeit:")
print(res_h1a_size.summary())
print(f"McFadden R2: {res_h1a_size.prsquared}")

# Likelihood-Ratio-Test gegen multinomialen Logit
y = df["SearchFound_scaled"]
X_mn = sm.add_constant(X_h1a_size)
mn_model = MNLogit(y, X_mn).fit(disp=0)
ll_ord = res_h1a_size.llf
ll_mn = mn_model.llf
lr_stat = 2 * (ll_mn - ll_ord)
df_diff = mn_model.df_model - res_h1a_size.df_model
p_val = chi2.sf(lr_stat, df_diff)
print(f"\nLR-Test vs. MNLogit: LR = {lr_stat:.2f}, p = {p_val:.4f}")

# -----------------------------------------------------------------------------  
# H1b: Parteilichkeit puffert Engagementverlust durch Shadowban  
# -----------------------------------------------------------------------------  
print("\n" + "="*70)
print("H1b: Shadowban × Parteilichkeit auf engagement_delta")
print("="*70)

model_h1b = smf.ols(
    "engagement_delta ~ C(SearchFound_scaled) * partisan_c + log_duration + log_SubscriberCount",
    data=df
).fit(cov_type="HC3")
print(model_h1b.summary())

# Auch auf Views (normalisierte Residuen)
model_views = smf.ols(
    "normalized_view_resiual_per_channel ~ SearchFound_scaled * partisan_c + log_duration + log_SubscriberCount",
    data=df
).fit(cov_type="HC3")
print("\nAbhängige Variable: normalisiertes View-Residual")
print(model_views.summary())

# Likes und Kommentare
model_likes = smf.ols(
    "like_delta ~ SearchFound_scaled * partisan_c + log_duration + log_SubscriberCount + log_description_words",
    data=df
).fit()
model_comments = smf.ols(
    "comment_delta ~ SearchFound_scaled * partisan_c + log_duration + log_SubscriberCount",
    data=df
).fit()
print("\nLike-Delta-Modell:")
print(model_likes.summary())
print("\nComment-Delta-Modell:")
print(model_comments.summary())

# -----------------------------------------------------------------------------  
# H2a: Interaktion zwischen Kanalgröße und Parteilichkeit  
# -----------------------------------------------------------------------------  
print("\n" + "="*70)
print("H2a: Ordered Logit mit Mega × partisan Interaktion")
print("="*70)

X_h2a = df[["Mega", "partisan_c", "mega_partisan", "partisan_c_sq",
            "log_duration", "engagement_rate_z", "log_channel_mean_views", "uploads_per_month"]]
model_h2a = OrderedModel(df["SearchFound_scaled"], X_h2a, distr="logit")
res_h2a = model_h2a.fit(method="bfgs")
print(res_h2a.summary())
print(f"McFadden R2: {res_h2a.prsquared}")

# -----------------------------------------------------------------------------  
# H2b: Größere Kanäle verlieren weniger Engagement durch Shadowban  
# -----------------------------------------------------------------------------  
print("\n" + "="*70)
print("H2b: Shadowban × log(SubscriberCount) auf engagement_delta")
print("="*70)

model_h2b = smf.ols(
    "engagement_delta ~ C(SearchFound_scaled) * log_SubscriberCount_c + log_duration",
    data=df
).fit(cov_type="HC3")
print(model_h2b.summary())

# Quadratische Terme prüfen
df["log_SubscriberCount_sq"] = df["log_SubscriberCount"] ** 2
df["log_duration_sq"] = df["log_duration"] ** 2
model_poly = smf.ols(
    "engagement_delta ~ C(SearchFound_scaled)*log_SubscriberCount + log_SubscriberCount_sq + log_duration + log_duration_sq",
    data=df
).fit(cov_type="HC3")
print("\nMit quadratischen Termen:")
print(model_poly.summary())

# -----------------------------------------------------------------------------  
# H3a & H3b: Topic-Diversität als Moderator  
# -----------------------------------------------------------------------------  
print("\n" + "="*70)
print("H3a: Ordered Logit mit Topic-Diversitätsinteraktion")
print("="*70)

X_h3 = df[["partisan_c", "avg_topic_diversity_c", "log_SubscriberCount_c",
           "log_duration", "engagement_rate_z", "partisan_c_sq"]].copy()
X_h3["partisan_x_diversity"] = X_h3["partisan_c"] * X_h3["avg_topic_diversity_c"]
model_h3 = OrderedModel(df["SearchFound_scaled"], X_h3, distr="logit")
res_h3 = model_h3.fit(method="bfgs")
print(res_h3.summary())
print(f"McFadden R2: {res_h3.prsquared}")

print("\nH3b: Topic-Diversität moderiert Shadowban → Engagementverlust")
model_h3b = smf.ols(
    "engagement_delta ~ C(SearchFound_scaled) * avg_topic_diversity_c + log_SubscriberCount_c + log_duration",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["channel_id"]})
print(model_h3b.summary())

# -----------------------------------------------------------------------------  
# Zusatzmodelle (Ghost Ban, binärer Logit)  
# -----------------------------------------------------------------------------  
print("\n" + "="*70)
print("Zusatz: Binärer Logit (shadowban_score_binary)")
print("="*70)

logit1 = smf.logit(
    "shadowban_score_binary ~ partisan_c + log_duration + title_length + log_SubscriberCount_c + log_engagement_rate + uploads_per_month",
    data=df
).fit()
print(logit1.summary())

logit2 = smf.logit(
    "shadowban_score_binary ~ partisan_c * log_SubscriberCount_c + log_duration + log_engagement_rate + uploads_per_month",
    data=df
).fit()
print(logit2.summary())

print("\nGhost ban model (negative view residual)")
ghost_model = smf.ols(
    "ghost_ban ~ partisan_c * log_SubscriberCount_c + log_duration + engagement_rate_z",
    data=df
).fit(cov_type="HC3")
print(ghost_model.summary())

# =============================================================================
# 9. ZUSÄTZLICHE MODELLE AUS DEM ZWEITEN CODE (BINÄRER LOGIT MIT KANALGRÖSSE)
# =============================================================================
print("\n" + "="*70)
print("Zusätzliche binäre Logit-Modelle mit Kanalgröße")
print("="*70)

logit3 = smf.logit(
    "shadowban_score_binary ~ C(channel_size) * partisan_c + log_duration + log_engagement_rate + normalized_view_resiual_per_channel",
    data=df
).fit()
print(logit3.summary())

logit4 = smf.logit(
    "shadowban_score_binary ~ partisan_c * log_SubscriberCount_c + log_duration + engagement_delta",
    data=df
).fit()
print(logit4.summary())

# =============================================================================
# 10. MODELLE MIT CLUSTER-ROBUSTEN STANDARDFEHLERN (KANALEBENE)
# =============================================================================
print("\n" + "="*70)
print("Modelle mit cluster-robusten Standardfehlern (channel_id)")
print("="*70)

# Modell für Views
model_cluster_views = smf.ols(
    "log_view_count ~ SearchFound_scaled * log_SubscriberCount + log_duration",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["channel_id"]})
print(model_cluster_views.summary())

# Modell für Engagement-Delta (bereits in H3b mit Cluster, aber wir wiederholen für Vollständigkeit)
model_cluster_eng = smf.ols(
    "engagement_delta ~ SearchFound_scaled * partisan_c + log_duration + log_SubscriberCount",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["channel_id"]})
print(model_cluster_eng.summary())

# =============================================================================
# 11. ABBILDUNGEN (zusätzlich)
# =============================================================================
# Hier können Sie weitere Visualisierungen einfügen, z.B. Verteilung der Shadowban-Scores
# oder Zusammenhänge zwischen Parteilichkeit und Shadowban.

plt.figure(figsize=(10,6))
sns.boxplot(x="SearchFound_scaled", y="partisan", data=df)
plt.title("Parteilichkeit nach Shadowban-Kategorie")
plt.xlabel("SearchFound_scaled")
plt.ylabel("Partisan")
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x="SearchFound_scaled", y="engagement_delta", hue="channel_size", data=df, alpha=0.5)
plt.title("Engagement-Abweichung vs. Shadowban-Score")
plt.xlabel("SearchFound_scaled")
plt.ylabel("Engagement Delta")
plt.legend(title="Channel Size")
plt.show()

print("\n" + "="*80)
print("ANALYSE ABGESCHLOSSEN")
print("="*80)

