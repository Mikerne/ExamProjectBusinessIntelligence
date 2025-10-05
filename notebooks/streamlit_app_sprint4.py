# streamlit_app_sprint4_userfriendly_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib, os

# Optional: SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="❤️ Heart Disease Explorer", layout="wide")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_data(path="../data/processed/heart_disease_clean.csv"):
    return pd.read_csv(path)

@st.cache_resource
def train_or_load_model(df, target='TenYearCHD', cache_path='rf_heart.joblib', use_cache=True):
    if use_cache and os.path.exists(cache_path):
        model = joblib.load(cache_path)
        scaler = joblib.load(cache_path + '.scaler')
        return model, scaler, True

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    joblib.dump(model, cache_path)
    joblib.dump(scaler, cache_path + '.scaler')
    return model, scaler, False

def intervals_to_str(df):
    """Konverter alle Interval-objekter i DataFrame til string for Plotly."""
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.IntervalDtype):
            df[col] = df[col].astype(str)
        else:
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, pd.Interval) else x)
    return df

# -------------------------
# Main app
# -------------------------
st.title("❤️ Heart Disease Explorer")
st.markdown("""
Denne app giver dig mulighed for at:
- Udforske datasættet
- Analysere mønstre i hjertesygdom (EDA)
- Træne & bruge en **Random Forest-model** til at forudsige risiko
- Visualisere clustering og PCA
- Afprøve patient-scenarier med individuelle prædiktioner
""")

# Load data
df = load_data()

# Tabs for navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data", 
    "🔎 EDA", 
    "🤖 Model & Prediction", 
    "🌀 Clustering", 
    "📈 Binned Analysis"
])

# -------------------------
# Tab 1: Dataset
# -------------------------
with tab1:
    st.subheader("Dataset preview & summary")
    st.dataframe(df.sample(min(500, len(df))).reset_index(drop=True))
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    st.write("**Statistisk oversigt:**")
    st.write(df.describe())

# -------------------------
# Tab 2: EDA
# -------------------------
with tab2:
    st.subheader("Exploratory Data Analysis")
    cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("#### 🔥 Korrelationsmatrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("#### 📦 Histogrammer")
    var = st.selectbox('Vælg variabel', cols)
    fig2 = px.histogram(intervals_to_str(df), x=var, nbins=30, marginal="box")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 🔄 Sammenligning med TenYearCHD")
    group_var = st.selectbox('Vælg variabel at gruppere', cols)
    df_plot = intervals_to_str(df)
    fig3 = px.histogram(
        df_plot,
        x=group_var,
        color='TenYearCHD',
        barmode='overlay',
        nbins=30
    )
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Tab 3: Model
# -------------------------
with tab3:
    st.subheader("Random Forest: Predict 10-year CHD")
    use_cache = st.checkbox("Brug gemt model, hvis tilgængelig", value=True)
    model, scaler, loaded = train_or_load_model(df, use_cache=use_cache)

    
    # Test metrics
    X = df.drop(columns=['TenYearCHD'])
    y = df['TenYearCHD']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_s = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)

    c1, c2 = st.columns(2)
    c1.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
    c2.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")

    with st.expander("Classification report"):
        st.text(classification_report(y_test, y_pred))

    st.markdown("#### 📌 Feature importance")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importances.head(15))

    st.markdown("#### 👤 Individuel patient-prædiktion")
    input_row = {}
    with st.form("predict_form"):
        for c in X.columns:
            input_row[c] = st.number_input(c, value=float(df[c].median()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        X_new = pd.DataFrame([input_row])
        X_new_s = scaler.transform(X_new)
        prob = model.predict_proba(X_new_s)[0, 1]

        # Vis kun risikoen
        if prob < 0.2:
            st.success(f"🟢 Lav risiko ({prob:.1%})")
        elif prob < 0.5:
            st.warning(f"🟡 Moderat risiko ({prob:.1%})")
        else:
            st.error(f"🔴 Høj risiko ({prob:.1%})")

        # SHAP forklaring, hvis tilgængelig
        if SHAP_AVAILABLE:
            st.markdown("#### SHAP forklaring")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_new_s)
            try:
                shap_html = shap.force_plot(
                    explainer.expected_value[1], shap_values[1], X_new, matplotlib=False
                )
                st.components.v1.html(shap_html.html(), height=400)
            except Exception:
                st.info("Kunne ikke vise SHAP plot i dette miljø")


# -------------------------
# Tab 4: Clustering
# -------------------------
with tab4:
    st.subheader("KMeans clustering & PCA (2D)")
    numeric = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric.drop(columns=['TenYearCHD'], errors="ignore"))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    cluster_k = st.slider("Antal clusters (KMeans)", 2, 6, 3)
    km = KMeans(n_clusters=cluster_k, random_state=42)
    labels = km.fit_predict(X_scaled)

    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=labels.astype(str),
                     title="PCA 2D clustering", hover_data=[df.index])
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Tab 5: Binned analysis
# -------------------------
with tab5:
    st.subheader("Binned analysis")
    numeric = df.select_dtypes(include=[np.number])
    bin_col = st.selectbox("Kolonne til binning", numeric.columns)
    agg_col = st.selectbox("Kolonne til gennemsnit", numeric.columns)
    nbins = st.slider("Antal bins", 3, 10, 5)

    df_bins = intervals_to_str(df.copy())
    df_bins['bins'] = pd.cut(df_bins[bin_col], bins=nbins).astype(str)
    agg = df_bins.groupby('bins', observed=True)[agg_col].mean().reset_index(name=f"mean_{agg_col}")

    fig = px.bar(
        agg,
        x='bins',
        y=f"mean_{agg_col}",
        title=f"Gennemsnit af {agg_col} pr. {bin_col}-interval"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.info("Kør appen lokalt med: `streamlit run streamlit_app_sprint4_userfriendly_fixed.py`")