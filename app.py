"""Streamlit dashboard for the UNSW-NB15 AutoML research prototype."""
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
META = RESULTS / "model_metadata.json"
METRICS = RESULTS / "final_model_metrics.json"
MODEL = ROOT / "artifacts" / "models" / "final_model.joblib"

st.set_page_config(page_title="AutoML | Network Defense Analytics", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at 12% 0%, #14233d 0, #0b1220 40%, #080d18 100%); color:#e6edf7}
[data-testid="stSidebar"] {background:#0a111e;border-right:1px solid #1e2b40}
[data-testid="stMetric"] {background:linear-gradient(145deg,#111e30,#0c1727);border:1px solid #20334d;padding:16px 18px;border-radius:14px}
[data-testid="stMetricLabel"] {color:#93a8c5}
[data-testid="stMetricValue"] {color:#f4f8ff}
.hero {padding:26px 30px;border:1px solid #233650;border-radius:18px;background:linear-gradient(115deg,rgba(18,35,58,.96),rgba(12,22,38,.92));margin-bottom:18px}
.eyebrow {text-transform:uppercase;letter-spacing:.16em;color:#67d9cc;font-size:.74rem;font-weight:700}
.hero h1 {margin:.35rem 0;color:#f4f8ff;font-size:2.25rem}
.hero p {color:#a9bad0;margin:0;max-width:850px}
.section-title {color:#f0f5ff;font-size:1.2rem;font-weight:650;margin:.3rem 0 .8rem}
.small-note {color:#93a8c5;font-size:.85rem}
div[data-testid="stFileUploader"] {border:1px dashed #3b5877;border-radius:14px;background:#0d1929;padding:12px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    meta = json.loads(META.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    if meta.get("model_type") == "autogluon":
        from autogluon.tabular import TabularPredictor
        model = TabularPredictor.load(meta["artifact"])
    else:
        model = joblib.load(MODEL)
    return model, meta, metrics

@st.cache_data
def csv_table(name):
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def prepare_upload(frame, required):
    forbidden = {"label", "attack_cat"} & set(frame.columns)
    clean = frame.drop(columns=list(forbidden)) if forbidden else frame.copy()
    missing = sorted(set(required) - set(clean.columns))
    if missing:
        raise ValueError("Required traffic columns are missing: " + ", ".join(missing))
    if clean.empty:
        raise ValueError("The uploaded CSV has no records.")
    return clean[list(required)].copy()

def probability_attack(model, X):
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X)
    return proba[1].to_numpy() if isinstance(proba, pd.DataFrame) else np.asarray(proba)[:, 1]

def show_image(file, caption=None):
    path = FIGURES / file
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)

def show_table(frame):
    """Render a compact HTML table without Streamlit's PyArrow conversion path."""
    st.markdown(frame.to_html(index=False, escape=True, classes="data-grid", border=0), unsafe_allow_html=True)

if not META.exists():
    st.error("Model metadata is missing. Run the training workflow first.")
    st.stop()

model, meta, metrics = load_bundle()
profile = json.loads((RESULTS / "dataset_profile.json").read_text(encoding="utf-8")) if (RESULTS / "dataset_profile.json").exists() else {}

with st.sidebar:
    st.markdown("### ◈ SENTINEL / LAB")
    st.caption("UNSW-NB15 · Research workspace")
    section = st.radio("WORKSPACE", ["Overview", "Live analysis", "Model comparison", "Dataset & bias", "Explainability"], label_visibility="collapsed")
    st.divider()
    st.markdown("**ACTIVE MODEL**")
    st.write(meta.get("model_name", "Research model"))
    st.caption(f"Validation F1 · {meta.get('validation_f1', float('nan')):.3f}")
    st.divider()
    st.caption("Research prototype only · Predictions are estimates, not proof of an attack.")

st.markdown(f"""<div class="hero"><div class="eyebrow">NETWORK TRAFFIC INTELLIGENCE · UNSW-NB15</div>
<h1>{section}</h1><p>Explore benchmark performance, inspect model behavior, or analyze a CSV of flow records. The selected model is <b>{meta.get('model_name','')}</b>, chosen using training-only validation evidence.</p></div>""", unsafe_allow_html=True)

if section == "Overview":
    train = profile.get("train", {}); test = profile.get("test", {})
    c = st.columns(5)
    c[0].metric("Selected model", meta.get("model_name", "—"))
    c[1].metric("Validation F1", f"{meta.get('validation_f1', 0):.3f}")
    c[2].metric("Test attack F1", f"{metrics.get('f1', 0):.3f}")
    c[3].metric("Attack recall", f"{metrics.get('recall', 0):.3f}")
    c[4].metric("Test ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="section-title">Experiment scorecard</div>', unsafe_allow_html=True)
        comp = csv_table("unified_comparison.csv")
        if not comp.empty and {"model", "f1"}.issubset(comp.columns):
            view = comp[[x for x in ["model", "validation_f1", "accuracy", "precision", "recall", "f1", "roc_auc", "search_seconds"] if x in comp.columns]].copy()
            show_table(view)
        else: st.info("Run reporting stage to build the comparison table.")
    with right:
        st.markdown('<div class="section-title">Attack vs normal · held-out test</div>', unsafe_allow_html=True)
        counts = pd.DataFrame({"Class":["Normal","Attack"],"Records":[test.get("label_counts",{}).get("0",test.get("label_counts",{}).get(0,0)),test.get("label_counts",{}).get("1",test.get("label_counts",{}).get(1,0))]})
        donut=alt.Chart(alt.Data(values=counts.to_dict("records"))).mark_arc(innerRadius=66).encode(theta=alt.Theta("Records:Q"),color=alt.Color("Class:N",scale=alt.Scale(domain=["Normal","Attack"],range=["#4f8cff","#ff6b78"])),tooltip=[alt.Tooltip("Class:N"),alt.Tooltip("Records:Q")])
        st.altair_chart(donut,use_container_width=True)
    st.markdown('<div class="section-title">Performance at a glance</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a: show_image("f1_comparison.png", "Test F1 across valid model experiments")
    with b: show_image("precision_recall_comparison.png", "Attack precision and recall")
    with c: show_image("final_confusion_matrix.png", "Selected model confusion matrix")

elif section == "Live analysis":
    st.markdown('<div class="section-title">Analyze a flow CSV</div>', unsafe_allow_html=True)
    st.caption("Upload up to 50,000 rows. Target fields are excluded from model inputs; missing required features are reported rather than invented.")
    samples = [("Balanced demo", "sample_upload.csv"), ("Attack-heavy demo", "sample_attack_heavy.csv"), ("Normal-heavy demo", "sample_normal_heavy.csv")]
    cols = st.columns(len(samples))
    for col, (label, filename) in zip(cols, samples):
        path = RESULTS / filename
        if path.exists(): col.download_button(f"Download {label} CSV", path.read_bytes(), filename, "text/csv", use_container_width=True)
    upload = st.file_uploader("Choose traffic CSV", type=["csv"])
    if upload:
        try:
            raw = pd.read_csv(upload, nrows=50001, low_memory=False)
            if len(raw) > 50000: raise ValueError("Upload exceeds the 50,000-row limit.")
            X = prepare_upload(raw, meta["input_features"])
            with st.spinner("Scoring flow records…"):
                pred = np.asarray(model.predict(X), dtype=int)
                prob = probability_attack(model, X)
            out = raw.copy(); out["prediction"] = np.where(pred == 1, "Potential attack", "Normal")
            if prob is not None: out["attack_probability"] = prob
            n_attack = int((pred == 1).sum()); n_normal = len(pred) - n_attack
            k = st.columns(4); k[0].metric("Flows analyzed", f"{len(pred):,}"); k[1].metric("Potential attacks", f"{n_attack:,}"); k[2].metric("Normal", f"{n_normal:,}"); k[3].metric("Attack share", f"{n_attack/max(1,len(pred)):.1%}")
            left,right = st.columns([1,1.25])
            with left:
                counts=pd.DataFrame({"Prediction":["Normal","Potential attack"],"Flows":[n_normal,n_attack]})
                bar=alt.Chart(alt.Data(values=counts.to_dict("records"))).mark_bar(cornerRadiusTopLeft=5,cornerRadiusTopRight=5).encode(x=alt.X("Prediction:N",sort=None),y=alt.Y("Flows:Q"),color=alt.Color("Prediction:N",scale=alt.Scale(domain=["Normal","Potential attack"],range=["#4f8cff","#ff6b78"])),tooltip=[alt.Tooltip("Prediction:N"),alt.Tooltip("Flows:Q")])
                st.altair_chart(bar,use_container_width=True)
            with right:
                if prob is not None:
                    hist_data=pd.DataFrame({"Attack probability":prob}); hist=alt.Chart(alt.Data(values=hist_data.to_dict("records"))).mark_bar(color="#5ed9c7").encode(x=alt.X("Attack probability:Q",bin=alt.Bin(maxbins=24)),y="count()")
                    st.altair_chart(hist,use_container_width=True)
            st.markdown("**Scored records**")
            st.dataframe(out.head(250),hide_index=True,use_container_width=True)
            st.download_button("Download scored CSV",out.to_csv(index=False).encode("utf-8"),"intrusion_predictions.csv","text/csv",type="primary")
        except Exception as exc: st.error(f"Could not analyze this CSV: {exc}")

elif section == "Model comparison":
    comp=csv_table("unified_comparison.csv")
    st.markdown('<div class="section-title">Compare valid experiments</div>',unsafe_allow_html=True)
    if not comp.empty:
        metric=st.selectbox("Chart metric",[x for x in ["f1","precision","recall","accuracy","roc_auc","validation_f1"] if x in comp.columns],format_func=lambda x:x.replace("_"," ").title())
        score=alt.Chart(alt.Data(values=comp.to_dict("records"))).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(x=alt.X("model:N",sort="-y",axis=alt.Axis(labelAngle=-25)),y=alt.Y(f"{metric}:Q",scale=alt.Scale(zero=False)),color=alt.Color("model:N",legend=None),tooltip=[alt.Tooltip("model:N")]+[alt.Tooltip(f"{x}:Q") for x in [metric,"validation_f1","precision","recall"] if x in comp.columns]).properties(title=f"{metric.replace('_',' ').title()} by model")
        st.altair_chart(score,use_container_width=True)
        if {"precision","recall"}.issubset(comp.columns):
            trade=alt.Chart(alt.Data(values=comp.to_dict("records"))).mark_circle(size=150).encode(x="precision:Q",y="recall:Q",color="model:N",tooltip=["model","precision","recall","f1"]).properties(title="Precision–recall tradeoff")
            st.altair_chart(trade,use_container_width=True)
        show_table(comp)
    st.markdown("**Search time vs test F1**"); show_image("search_time_vs_f1.png")
    st.markdown("**Selected model confusion matrix**"); show_image("final_confusion_matrix.png")
    st.caption("Search time is not a standardized measure of hardware use; the frameworks use different search strategies and resource behavior.")

elif section == "Dataset & bias":
    train=profile.get("train",{}); test=profile.get("test",{})
    cols=st.columns(4); cols[0].metric("Train records",f"{train.get('shape',[0])[0]:,}"); cols[1].metric("Test records",f"{test.get('shape',[0])[0]:,}"); cols[2].metric("Train features",str(train.get("shape",[0,0])[1]-3)); cols[3].metric("Exact cross-partition duplicates",str(profile.get("exact_duplicate_feature_rows_across_partitions","—")))
    p1,p2=st.columns(2)
    with p1: show_image("train_label_distribution.png","Training label counts"); show_image("train_attack_categories.png","Training attack categories")
    with p2: show_image("test_label_distribution.png","Test label counts"); show_image("test_attack_categories.png","Test attack categories")
    st.markdown("**Numeric feature distributions and correlation**")
    p1,p2=st.columns(2)
    with p1: show_image("numeric_distributions.png")
    with p2: show_image("correlation_heatmap.png")
    shift=profile.get("numeric_distribution_shift",{})
    if shift:
        shift_df=pd.DataFrame([{"Feature":k,**v} for k,v in shift.items()]).sort_values("ks_statistic",ascending=False)
        st.markdown("**Largest numeric train/test shifts (KS statistic)**")
        show_table(shift_df.head(15))
    st.info("Class prevalence differs between supplied partitions. Rare attack categories have limited support. This is a class-representation and collection-bias analysis, not demographic fairness analysis.")

else:
    st.markdown('<div class="section-title">How the model behaves</div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a: show_image("shap_summary.png","SHAP summary · Random Forest surrogate")
    with b: show_image("shap_global_importance.png","Global mean absolute SHAP value")
    local_path=RESULTS/"shap_local_explanation.json"
    if local_path.exists():
        local=json.loads(local_path.read_text(encoding="utf-8")); st.markdown(f"**Local example:** predicted class {local.get('predicted_class')} · {'surrogate explanation' if local.get('is_surrogate') else 'deployed model explanation'}")
        st.caption(local.get("contribution_interpretation","SHAP describes model behavior, not causation."))
        show_table(pd.DataFrame(local.get("top_contributions",[])))
    st.warning("SHAP values explain the Random Forest surrogate, not the deployed AutoGluon ensemble. They are not causal explanations.")

st.divider()
st.caption(f"Model version 1.0 · Target: {meta.get('target','label (0=Normal, 1=Attack)')} · Research estimate only; not validated for production network defense.")
