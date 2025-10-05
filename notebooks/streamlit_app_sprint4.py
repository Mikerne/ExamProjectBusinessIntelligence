# streamlit_app_sprint4.py
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
import joblib
import os

# Optional: SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="Heart Disease Explorer", layout="wide")

# -------------------------
# Helper functions
# -------------------------

def load_data(path="../data/processed/heart_disease_clean.csv"):
    df = pd.read_csv(path)
    return df


def train_or_load_model(df, target='TenYearCHD', cache_path='rf_heart.joblib'):
    if os.path.exists(cache_path):
        model = joblib.load(cache_path)
        scaler = joblib.load(cache_path + '.scaler')
        return model, scaler, True

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(X_train_s, y_train)

    # Evaluate and save
    y_proba = model.predict_proba(X_test_s)[:,1]
    y_pred = model.predict(X_test_s)
    print('ROC AUC', roc_auc_score(y_test, y_proba))
    joblib.dump(model, cache_path)
    joblib.dump(scaler, cache_path + '.scaler')
    return model, scaler, False


# -------------------------
# Main app layout
# -------------------------

st.title("Sprint 4 — Business Application: Heart Disease Explorer")

# Load data
with st.spinner('Loading data...'):
    df = load_data()

# Sidebar controls
with st.sidebar:
    st.header('Controls')
    show_preview = st.checkbox('Show dataset preview', value=True)
    show_eda = st.checkbox('Show EDA', value=True)
    show_model = st.checkbox('Show Model & Prediction', value=True)
    use_saved_model = st.checkbox('Use cached model if available', value=True)
    if SHAP_AVAILABLE:
        show_shap = st.checkbox('Show SHAP explanations', value=True)
    else:
        st.write('SHAP not installed — explanations disabled')
    st.markdown('---')
    st.write('Deployment notes: run `streamlit run streamlit_app_sprint4.py`')

# Dataset preview
if show_preview:
    st.subheader('Dataset preview & summary')
    st.dataframe(df.sample(min(500, len(df))).reset_index(drop=True))
    st.write('Shape:', df.shape)
    st.write(df.describe())

# EDA
if show_eda:
    st.subheader('Exploratory Data Analysis')
    cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown('**Correlation Heatmap**')
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown('**Histograms (select variable)**')
    var = st.selectbox('Choose variable for histogram', cols, index=0)
    fig2 = px.histogram(df, x=var, nbins=30, title=f'Histogram of {var}', marginal='box')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('**Grouped Histograms: TenYearCHD**')
    group_var = st.selectbox('Variable to group by TenYearCHD', cols, index=min(3, len(cols)-1))
    fig3 = px.histogram(df, x=group_var, color='TenYearCHD', barmode='overlay', nbins=30)
    st.plotly_chart(fig3, use_container_width=True)

# Model training / load
if show_model:
    st.subheader('Model: Random Forest (predict 10-year CHD)')

    model, scaler, loaded = train_or_load_model(df, target='TenYearCHD', cache_path='rf_heart.joblib' if use_saved_model else 'rf_heart_temp.joblib')
    if loaded:
        st.success('Loaded cached model')
    else:
        st.info('Trained a new model and cached it')

    # Show metrics on holdout sample
    X = df.drop(columns=['TenYearCHD'])
    y = df['TenYearCHD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_s = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_s)[:,1]
    y_pred = model.predict(X_test_s)
    st.write('ROC AUC (test):', roc_auc_score(y_test, y_proba))
    st.write('Accuracy (test):', accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))

    # Feature importance
    try:
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.markdown('**Top features (importance)**')
        st.bar_chart(importances.head(15))
    except Exception:
        st.write('Feature importance unavailable')

    # Interactive prediction
    st.markdown('---')
    st.markdown('### Predict individual risk')
    input_row = {}
    cols_for_input = X.columns.tolist()
    with st.form('predict_form'):
        st.write('Fill patient attributes (use realistic ranges)')
        for c in cols_for_input:
            val = st.number_input(c, value=float(df[c].median()))
            input_row[c] = val
        submitted = st.form_submit_button('Predict')
        if submitted:
            X_new = pd.DataFrame([input_row])
            X_new_s = scaler.transform(X_new)
            prob = model.predict_proba(X_new_s)[0,1]
            st.metric('Predicted 10-year CHD probability', f"{prob:.3f}")

            # SHAP explanation
            if SHAP_AVAILABLE and show_shap:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_new_s)
                st.write('SHAP explanation (force plot)')
                try:
                    shap_html = shap.force_plot(explainer.expected_value[1], shap_values[1], X_new, matplotlib=False)
                    st.components.v1.html(shap_html.html(), height=400)
                except Exception:
                    st.write('Could not render SHAP force plot in this environment')

# Clustering visualization
st.subheader('Clustering & PCA (2D)')

numeric = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric.drop(columns=['TenYearCHD'], errors='ignore'))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
cluster_k = st.slider('Choose number of clusters (KMeans)', 2, 6, 3)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=cluster_k, random_state=42)
labels = km.fit_predict(X_scaled)

fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=labels.astype(str), hover_data=[df.index], title='PCA 2D clustering')
st.plotly_chart(fig, use_container_width=True)

# Binned analysis
st.subheader('Binned analysis example')
bin_col = st.selectbox('Column to bin', numeric.columns.tolist(), index=0)
agg_col = st.selectbox('Column to aggregate', numeric.columns.tolist(), index=min(1, len(numeric.columns)-1))
nbins = st.slider('Number of bins', 3, 10, 5)

bins = pd.cut(df[bin_col], bins=nbins)
agg = df.groupby(bins)[agg_col].mean().reset_index()
fig = px.bar(agg, x=agg_col, y=agg_col, title=f'Mean {agg_col} per {bin_col} bin')
st.plotly_chart(fig)

# Footer: deployment & evaluation guidance
st.markdown('---')
st.header('Deployment & Usability')
st.markdown(
"""
**Run locally:**
- Install dependencies: `pip install -r requirements.txt` (see README)
- `streamlit run streamlit_app_sprint4.py`
"""
)

st.write('End of app')
