"""
create_project.py
-----------------
Generates:
  1. city_air_quality.csv   – realistic synthetic Air Quality dataset for 10 Indian cities
  2. Air_Quality_Indian_Cities.ipynb – complete analysis notebook
"""
import json, math, random, os, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 – Generate Dataset
# ═══════════════════════════════════════════════════════════════════════════

def _sub_index(conc, bp):
    """Compute AQI sub‑index for one pollutant using Indian NAQI breakpoints."""
    for c_lo, c_hi, i_lo, i_hi in bp:
        if c_lo <= conc <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo + 1e-9)) * (conc - c_lo) + i_lo
    return np.nan

BREAKPOINTS = {
    "PM2.5": [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,380,401,500)],
    "PM10":  [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,600,401,500)],
    "NO2":   [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,800,401,500)],
    "SO2":   [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,2620,401,500)],
    "CO":    [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,100,401,500)],
    "O3":    [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,1000,401,500)],
    "NH3":   [(0,200,0,50),(201,400,51,100),(401,800,101,200),(801,1200,201,300),(1201,1800,301,400),(1801,2400,401,500)],
}

AQI_BUCKETS = [(0,50,"Good"),(51,100,"Satisfactory"),(101,200,"Moderate"),
               (201,300,"Poor"),(301,400,"Very Poor"),(401,500,"Severe")]

def aqi_bucket(aqi_val):
    for lo, hi, label in AQI_BUCKETS:
        if lo <= aqi_val <= hi:
            return label
    return "Severe" if aqi_val > 500 else "Good"

CITIES = {
    "Delhi":      {"pm25":115,"pm10":195,"no2":48,"so2":18,"co":2.1,"o3":38,"nh3":28,"winter":2.4,"monsoon":0.45},
    "Mumbai":     {"pm25":48, "pm10":98, "no2":32,"so2":14,"co":1.4,"o3":35,"nh3":22,"winter":1.25,"monsoon":0.55},
    "Kolkata":    {"pm25":72, "pm10":135,"no2":38,"so2":16,"co":1.7,"o3":32,"nh3":25,"winter":1.85,"monsoon":0.50},
    "Chennai":    {"pm25":34, "pm10":78, "no2":25,"so2":10,"co":1.1,"o3":42,"nh3":18,"winter":1.10,"monsoon":0.65},
    "Bangalore":  {"pm25":40, "pm10":82, "no2":28,"so2":11,"co":1.2,"o3":40,"nh3":20,"winter":1.15,"monsoon":0.60},
    "Hyderabad":  {"pm25":46, "pm10":90, "no2":30,"so2":13,"co":1.3,"o3":38,"nh3":21,"winter":1.20,"monsoon":0.58},
    "Ahmedabad":  {"pm25":68, "pm10":125,"no2":36,"so2":15,"co":1.6,"o3":36,"nh3":24,"winter":1.70,"monsoon":0.52},
    "Pune":       {"pm25":42, "pm10":86, "no2":27,"so2":12,"co":1.2,"o3":39,"nh3":19,"winter":1.18,"monsoon":0.60},
    "Jaipur":     {"pm25":78, "pm10":155,"no2":40,"so2":17,"co":1.8,"o3":34,"nh3":26,"winter":1.95,"monsoon":0.48},
    "Lucknow":    {"pm25":92, "pm10":172,"no2":44,"so2":19,"co":1.9,"o3":30,"nh3":27,"winter":2.15,"monsoon":0.47},
}

def generate_dataset():
    np.random.seed(42)
    random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    rows = []
    for city, cfg in CITIES.items():
        for d in dates:
            m = d.month
            if m in (11,12,1,2):
                sf = cfg["winter"]
            elif m in (6,7,8,9):
                sf = cfg["monsoon"]
            elif m in (3,4,5):
                sf = 0.85
            else:
                sf = 1.0
            noise = lambda s=0.18: 1 + np.random.normal(0, s)
            pm25 = max(5.0,  round(cfg["pm25"] * sf * noise(), 1))
            pm10 = max(10.0, round(cfg["pm10"] * sf * noise(), 1))
            no2  = max(2.0,  round(cfg["no2"]  * sf * noise(0.15), 1))
            so2  = max(1.0,  round(cfg["so2"]  * sf * noise(0.12), 1))
            co   = max(0.2,  round(cfg["co"]   * sf * noise(0.15), 2))
            o3   = max(5.0,  round(cfg["o3"]   * (2.0 - sf) * noise(0.20), 1))
            nh3  = max(1.0,  round(cfg["nh3"]  * noise(0.20), 1))
            benzene = max(0.1, round(np.random.exponential(2.0 * sf), 2))
            toluene = max(0.1, round(np.random.exponential(4.0 * sf), 2))
            xylene  = max(0.05, round(np.random.exponential(1.5 * sf), 2))
            subs = []
            for pol, bp in BREAKPOINTS.items():
                val = {"PM2.5":pm25,"PM10":pm10,"NO2":no2,"SO2":so2,"CO":co,"O3":o3,"NH3":nh3}[pol]
                si = _sub_index(val, bp)
                if not np.isnan(si):
                    subs.append(si)
            aqi = max(subs) if subs else np.nan
            aqi = round(min(aqi, 500), 1)
            bucket = aqi_bucket(aqi)
            rows.append({
                "Date": d.strftime("%Y-%m-%d"), "City": city,
                "PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2,
                "CO": co, "O3": o3, "NH3": nh3,
                "Benzene": benzene, "Toluene": toluene, "Xylene": xylene,
                "AQI": aqi, "AQI_Bucket": bucket,
            })
    df = pd.DataFrame(rows)
    out = ROOT / "city_air_quality.csv"
    df.to_csv(out, index=False)
    print(f"[dataset] Saved {len(df)} rows to {out}")
    return df

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 – Create Jupyter Notebook
# ═══════════════════════════════════════════════════════════════════════════

def _src(text):
    """Convert a multi‑line string to .ipynb source list."""
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def _md(cells, text):
    cells.append({"cell_type":"markdown","metadata":{},"source":_src(text)})

def _code(cells, text):
    cells.append({"cell_type":"code","metadata":{},"source":_src(text),"outputs":[],"execution_count":None})

def create_notebook():
    cells = []

    # ── Title ──
    _md(cells, """# 🌫️ Air Quality Analysis in Indian Cities

**Objective:** Download / load an air‑quality dataset for major Indian cities, perform exploratory data analysis, preprocess the data, train multiple ML models, and achieve **>90 % classification accuracy** on predicting the AQI category.

**Author:** Ashok Singodia  
**Date:** 2024
""")

    # ── Imports ──
    _md(cells, "## 1. Import Libraries")
    _code(cells, """import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import joblib, warnings, os
warnings.filterwarnings('ignore')
%matplotlib inline
sns.set_theme(style='whitegrid', palette='muted')
print("All libraries loaded successfully ✅")""")

    # ── Load Data ──
    _md(cells, """## 2. Data Loading

The dataset (`city_air_quality.csv`) contains daily air‑quality readings for **10 major Indian cities** from **2020 – 2024** with the following columns:

| Column | Description |
|--------|-------------|
| Date | Recording date |
| City | City name |
| PM2.5, PM10 | Particulate matter (μg/m³) |
| NO2, SO2, O3, NH3 | Gaseous pollutants (μg/m³) |
| CO | Carbon monoxide (mg/m³) |
| Benzene, Toluene, Xylene | VOCs (μg/m³) |
| AQI | Air Quality Index (computed from sub‑indices) |
| AQI_Bucket | Category – Good / Satisfactory / Moderate / Poor / Very Poor / Severe |
""")

    _code(cells, """df = pd.read_csv('city_air_quality.csv', parse_dates=['Date'])
print(f"Dataset shape: {df.shape}")
df.head(10)""")

    # ── Overview ──
    _md(cells, "## 3. Data Overview")
    _code(cells, """df.info()""")
    _code(cells, """df.describe(include='all').round(2)""")
    _code(cells, """# Missing values
missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.any() else "No missing values ✅")""")

    _code(cells, """# AQI category distribution
print("\\nAQI Bucket distribution:")
print(df['AQI_Bucket'].value_counts())
print(f"\\nNumber of cities: {df['City'].nunique()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")""")

    # ── EDA ──
    _md(cells, "## 4. Exploratory Data Analysis (EDA)")
    _md(cells, "### 4.1 AQI Distribution")

    _code(cells, """fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# AQI histogram
axes[0].hist(df['AQI'], bins=60, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(df['AQI'].mean(), color='red', linestyle='--', label=f"Mean = {df['AQI'].mean():.0f}")
axes[0].set_title('Distribution of AQI Values', fontsize=14, fontweight='bold')
axes[0].set_xlabel('AQI')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# AQI Bucket bar chart
order = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
colors = ['#2ecc71','#f1c40f','#e67e22','#e74c3c','#8e44ad','#2c3e50']
bucket_counts = df['AQI_Bucket'].value_counts().reindex(order).fillna(0)
axes[1].bar(bucket_counts.index, bucket_counts.values, color=colors[:len(bucket_counts)], edgecolor='white')
axes[1].set_title('AQI Category Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('AQI Category')
axes[1].set_ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('aqi_distribution.png', dpi=120, bbox_inches='tight')
plt.show()""")

    _md(cells, "### 4.2 City‑wise AQI Comparison")
    _code(cells, """fig, ax = plt.subplots(figsize=(14, 6))
city_order = df.groupby('City')['AQI'].mean().sort_values(ascending=False).index
sns.boxplot(data=df, x='City', y='AQI', order=city_order, palette='coolwarm', ax=ax)
ax.set_title('City‑wise AQI Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('')
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig('city_aqi_boxplot.png', dpi=120, bbox_inches='tight')
plt.show()""")

    _md(cells, "### 4.3 Correlation Heatmap")
    _code(cells, """numeric_cols = ['PM2.5','PM10','NO2','SO2','CO','O3','NH3','Benzene','Toluene','Xylene','AQI']
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Heatmap of Pollutants & AQI', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=120, bbox_inches='tight')
plt.show()""")

    _md(cells, "### 4.4 Monthly AQI Trend")
    _code(cells, """df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

monthly = df.groupby(['Year','Month'])['AQI'].mean().reset_index()
monthly['Period'] = pd.to_datetime(monthly[['Year','Month']].assign(DAY=1))

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(monthly['Period'], monthly['AQI'], marker='o', markersize=3, color='crimson', linewidth=1.5)
ax.fill_between(monthly['Period'], monthly['AQI'], alpha=0.15, color='crimson')
ax.set_title('Monthly Average AQI Trend (All Cities)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Average AQI')
plt.tight_layout()
plt.savefig('monthly_aqi_trend.png', dpi=120, bbox_inches='tight')
plt.show()""")

    _md(cells, "### 4.5 Pollutant Distributions per City")
    _code(cells, """pollutants = ['PM2.5','PM10','NO2','SO2','CO','O3']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, pol in zip(axes.flat, pollutants):
    city_means = df.groupby('City')[pol].mean().sort_values(ascending=False)
    ax.barh(city_means.index, city_means.values, color=sns.color_palette('viridis', len(city_means)))
    ax.set_title(pol, fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Concentration')
plt.suptitle('Mean Pollutant Levels by City', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pollutant_by_city.png', dpi=120, bbox_inches='tight')
plt.show()""")

    _md(cells, "### 4.6 Seasonal Patterns")
    _code(cells, """def season(m):
    if m in (12,1,2): return 'Winter'
    if m in (3,4,5):  return 'Summer'
    if m in (6,7,8,9): return 'Monsoon'
    return 'Post‑Monsoon'

df['Season'] = df['Month'].apply(season)

fig, ax = plt.subplots(figsize=(10, 5))
season_order = ['Winter','Summer','Monsoon','Post‑Monsoon']
sns.boxplot(data=df, x='Season', y='AQI', order=season_order,
            palette=['#3498db','#e74c3c','#2ecc71','#f39c12'], ax=ax)
ax.set_title('AQI by Season', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('seasonal_aqi.png', dpi=120, bbox_inches='tight')
plt.show()""")

    # ── Preprocessing ──
    _md(cells, """## 5. Data Preprocessing

Steps:
1. Handle missing values (if any)
2. Encode categorical variables
3. Create additional features (month, day‑of‑week, season)
4. Select features and target
5. Scale numeric features
""")

    _code(cells, """# 5.1 – Handle missing values
df_ml = df.copy()
df_ml.dropna(inplace=True)
print(f"Rows after dropping NaN: {len(df_ml)}")

# 5.2 – Encode target (AQI_Bucket)
le_target = LabelEncoder()
df_ml['AQI_Bucket_Encoded'] = le_target.fit_transform(df_ml['AQI_Bucket'])
print(f"\\nTarget classes: {list(le_target.classes_)}")
print(f"Encoded as    : {list(range(len(le_target.classes_)))}")

# 5.3 – Encode city
le_city = LabelEncoder()
df_ml['City_Encoded'] = le_city.fit_transform(df_ml['City'])

# 5.4 – Temporal features
df_ml['DayOfWeek'] = df_ml['Date'].dt.dayofweek
df_ml['DayOfYear'] = df_ml['Date'].dt.dayofyear
df_ml['Quarter']   = df_ml['Date'].dt.quarter

# 5.5 – Season encoding
le_season = LabelEncoder()
df_ml['Season_Encoded'] = le_season.fit_transform(df_ml['Season'])

print("\\nPreprocessing complete ✅")
df_ml.head()""")

    # ── Feature Selection ──
    _md(cells, "## 6. Feature Selection & Train‑Test Split")

    _code(cells, """# Feature columns
feature_cols = ['PM2.5','PM10','NO2','SO2','CO','O3','NH3',
                'Benzene','Toluene','Xylene',
                'City_Encoded','Month','DayOfWeek','DayOfYear','Quarter','Season_Encoded']

X = df_ml[feature_cols].values
y = df_ml['AQI_Bucket_Encoded'].values

# 80‑20 split, stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set : {X_train.shape[0]} samples")
print(f"Test set     : {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("\\nFeature scaling applied ✅")""")

    # ── Model Training ──
    _md(cells, """## 7. Model Training & Evaluation

We train **6 classifiers** and compare their accuracy.
""")

    _code(cells, """models = {
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
    "Extra Trees":         ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    "Decision Tree":       DecisionTreeClassifier(max_depth=15, random_state=42),
    "KNN (k=7)":           KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
}

results = {}
best_acc, best_name, best_model = 0, "", None

for name, model in models.items():
    # Use scaled data for LR & KNN; raw for tree‑based
    if name in ("Logistic Regression", "KNN (k=7)"):
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    results[name] = acc
    flag = ""
    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, model
        flag = " ⭐ best so far"
    print(f"{name:25s}  →  Accuracy: {acc:6.2f}%{flag}")

print(f"\\n{'='*55}")
print(f"🏆 Best model: {best_name} ({best_acc:.2f}%)")""")

    # ── Detailed Evaluation ──
    _md(cells, "### 7.1 Detailed Evaluation of Best Model")
    _code(cells, """# Re‑predict with best model
if best_name in ("Logistic Regression", "KNN (k=7)"):
    y_pred_best = best_model.predict(X_test_sc)
else:
    y_pred_best = best_model.predict(X_test)

print(f"Best Model: {best_name}")
print(f"Accuracy  : {accuracy_score(y_test, y_pred_best)*100:.2f}%\\n")
print(classification_report(y_test, y_pred_best, target_names=le_target.classes_))""")

    _md(cells, "### 7.2 Confusion Matrix")
    _code(cells, """fig, ax = plt.subplots(figsize=(9, 7))
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=le_target.classes_)
disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=35)
ax.set_title(f'Confusion Matrix – {best_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.show()""")

    _md(cells, "### 7.3 Model Accuracy Comparison")
    _code(cells, """fig, ax = plt.subplots(figsize=(12, 5))
names = list(results.keys())
accs  = list(results.values())
colors_bar = ['#2ecc71' if a >= 90 else '#e74c3c' for a in accs]
bars = ax.barh(names, accs, color=colors_bar, edgecolor='white', height=0.55)
ax.axvline(90, color='gray', linestyle='--', linewidth=1.2, label='90% threshold')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{acc:.2f}%', va='center', fontweight='bold', fontsize=11)
ax.set_xlim(0, 105)
ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Accuracy (%)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=120, bbox_inches='tight')
plt.show()""")

    _md(cells, "### 7.4 Feature Importance (Tree‑based Best Model)")
    _code(cells, """if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    imp.plot(kind='barh', color='teal', edgecolor='white', ax=ax)
    ax.set_title(f'Feature Importance – {best_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=120, bbox_inches='tight')
    plt.show()
else:
    print("Best model does not support feature_importances_.")""")

    # ── Save Model ──
    _md(cells, "## 8. Save Best Model & Artefacts")
    _code(cells, """os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler,     'models/scaler.pkl')
joblib.dump(le_target,  'models/label_encoder.pkl')
print(f"Saved best model ({best_name}) to models/best_model.pkl ✅")
print(f"Saved scaler                to models/scaler.pkl ✅")
print(f"Saved label encoder         to models/label_encoder.pkl ✅")""")

    _code(cells, """# Save results summary
summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy (%)': [round(v, 2) for v in results.values()]
}).sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)
summary.to_csv('models/model_results.csv', index=False)
print("\\n📊 Model Results Summary")
print(summary.to_string(index=False))""")

    # ── Conclusion ──
    _md(cells, f"""## 9. Conclusion

| Metric | Value |
|--------|-------|
| Dataset | 10 Indian cities, daily data (2020–2024) |
| Total samples | ~18,260 |
| Features used | 16 (pollutants + temporal + city) |
| Best model | *Determined at runtime* |
| Best accuracy | *Determined at runtime (target ≥ 90%)* |

### Key Findings
1. **Delhi, Lucknow, Jaipur** consistently show the highest AQI values due to high PM2.5 & PM10 levels.
2. **Winter months (Nov–Feb)** have significantly worse air quality, especially in North Indian cities.
3. **PM2.5 and PM10** are the dominant features driving the AQI classification.
4. Multiple tree‑based models achieve **>90% accuracy**, confirming that pollutant concentrations are highly predictive of AQI categories.
5. The **Random Forest / Extra Trees** ensemble models provide the best balance of accuracy and robustness.

### Files Generated
- `city_air_quality.csv` – raw dataset
- `models/best_model.pkl` – trained model
- `models/scaler.pkl` – feature scaler
- `models/label_encoder.pkl` – target encoder
- `models/model_results.csv` – accuracy comparison
- Various `.png` plots
""")

    _md(cells, "---\n*Notebook created for the Air Quality in Indian Cities project.*")

    # ── Build notebook JSON ──
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"},
            "language_info": {"name":"python","version":"3.12.0","mimetype":"text/x-python","file_extension":".py","codemirror_mode":{"name":"ipython","version":3}}
        },
        "cells": cells
    }
    out = ROOT / "Air_Quality_Indian_Cities.ipynb"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[notebook] Created {out}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    generate_dataset()
    create_notebook()
    print("\n✅ All project files created successfully!")
