import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os, pickle, json, requests
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Model Fairness Audit Platform",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.main .block-container{padding:2rem 2.5rem;max-width:1350px;}

.step-card{background:#161b22;border:1px solid #21262d;border-radius:14px;padding:2rem;margin-bottom:1.2rem;}
.step-card.active{border-color:#238636;box-shadow:0 0 0 2px #23863640;}
.step-card.done{border-color:#1f6feb;}
.step-card.locked{opacity:0.45;}
.step-badge{display:inline-block;font-family:'Space Mono',monospace;font-size:0.7rem;
    letter-spacing:2px;text-transform:uppercase;padding:3px 12px;border-radius:20px;margin-bottom:0.6rem;}
.step-title{font-family:'Space Mono',monospace;font-size:1.15rem;font-weight:700;color:#e6edf3;margin-bottom:0.3rem;}
.step-sub{font-size:0.85rem;color:#8b949e;}
.done-tag{font-family:'Space Mono',monospace;font-size:0.72rem;color:#56d364;margin-top:0.6rem;}

/* Tab-style option cards */
.opt-card{
    background:#0d1117;border:2px solid #21262d;border-radius:12px;
    padding:1.5rem;cursor:pointer;transition:border-color 0.2s;
}
.opt-card.selected{border-color:#238636;background:#0d1f12;}
.opt-card-title{font-family:'Space Mono',monospace;font-size:0.9rem;
    font-weight:700;color:#e6edf3;margin-bottom:0.3rem;}
.opt-card-sub{font-size:0.82rem;color:#8b949e;}

.mcard{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:1.2rem;text-align:center;}
.mval{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;color:#58a6ff;}
.mval.g{color:#56d364;}.mval.o{color:#ffa657;}.mval.r{color:#f85149;}
.mlbl{font-size:0.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-top:4px;}

.ibox{background:#0d1117;border:1px solid #21262d;border-radius:8px;
    padding:1rem 1.4rem;font-family:'Space Mono',monospace;font-size:0.8rem;color:#8b949e;line-height:2;}
.shdr{font-family:'Space Mono',monospace;font-size:0.78rem;color:#8b949e;text-transform:uppercase;
    letter-spacing:2px;border-bottom:1px solid #21262d;padding-bottom:8px;margin:1.5rem 0 1rem;}

.v-fair{background:#1a4731;border:1px solid #238636;border-radius:10px;padding:1rem 1.5rem;
    color:#56d364;font-weight:700;font-size:1.05rem;text-align:center;margin:1rem 0;}
.v-mild{background:#3d1f00;border:1px solid #d4800a;border-radius:10px;padding:1rem 1.5rem;
    color:#ffa657;font-weight:700;font-size:1.05rem;text-align:center;margin:1rem 0;}
.v-bias{background:#3d0000;border:1px solid #f85149;border-radius:10px;padding:1rem 1.5rem;
    color:#f85149;font-weight:700;font-size:1.05rem;text-align:center;margin:1rem 0;}

/* API key input styling */
.api-box{background:#0d1117;border:1px solid #238636;border-radius:10px;padding:1.5rem;}

div[data-testid="stButton"]>button{
    background:#238636;color:#fff;border:none;border-radius:8px;
    font-weight:600;font-size:0.95rem;padding:0.6rem 1.5rem;width:100%;}
div[data-testid="stButton"]>button:hover{background:#2ea043;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "model_obj":     None,
    "model_name":    "",
    "model_ready":   False,
    "model_source":  "",   # "upload" or "api"
    "api_endpoint":  "",
    "api_key":       "",
    "api_format":    "json",
    "df_test":       None,
    "target_col":    "",
    "sensitive_col": "",
    "data_ready":    False,
    "audit_done":    False,
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
def fix_sklearn_model_compatibility(model):
    """
    Fix sklearn pickle compatibility issues between different sklearn versions.
    Some older DecisionTree/RandomForest models do not contain monotonic_cst.
    """
    if not hasattr(model, "monotonic_cst"):
        try:
            model.monotonic_cst = None
        except:
            pass

    if hasattr(model, "estimators_"):
        try:
            for est in model.estimators_:
                if not hasattr(est, "monotonic_cst"):
                    est.monotonic_cst = None
        except:
            pass

    return model

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #21262d;
     border-radius:14px;padding:2rem 2.5rem;margin-bottom:2rem;border-top:3px solid #238636;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem">
    <div>
      <h1 style="font-family:Space Mono,monospace;font-size:2rem;color:#e6edf3;margin:0">
        ⚖️ AI Model Fairness Audit Platform
      </h1>
      <p style="color:#8b949e;margin:0.4rem 0 0.8rem;font-size:0.9rem">
        Fairlearn Bias Detection Tool — Upload Model or Connect via API · Test Data · Run Audit
      </p>
      <span style="background:#1a4731;color:#56d364;border:1px solid #238636;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:600">OWASP ML Top 10</span>
      <span style="background:#1c2c4a;color:#58a6ff;border:1px solid #1f6feb;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:600;margin-left:6px">NIST AI RMF</span>
      <span style="background:#3d1f00;color:#ffa657;border:1px solid #d4800a;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:600;margin-left:6px">MITRE ATLAS</span>
    </div>
    <div style="font-family:Space Mono,monospace;font-size:11px;color:#8b949e;text-align:right">
      Powered by<br>
      <span style="color:#58a6ff;font-size:14px">Microsoft Fairlearn</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── STEP PROGRESS CARDS ───────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

c1.markdown(f"""
<div class="step-card {'active' if not st.session_state.model_ready else 'done'}">
  <div class="step-badge" style="background:#1a4731;color:#56d364;border:1px solid #238636">
    {"✅  COMPLETE" if st.session_state.model_ready else "🔵  STEP 1"}
  </div>
  <div class="step-title">Connect Model</div>
  <div class="step-sub">Upload .pkl file OR connect via API endpoint</div>
  {"<div class='done-tag'>✓ " + st.session_state.model_name + " (" + st.session_state.model_source.upper() + ")</div>" if st.session_state.model_ready else ""}
</div>""", unsafe_allow_html=True)

s2 = "done" if st.session_state.data_ready else ("active" if st.session_state.model_ready else "locked")
c2.markdown(f"""
<div class="step-card {s2}">
  <div class="step-badge" style="background:{'#1a4731' if st.session_state.data_ready else '#1c2c4a'};
       color:{'#56d364' if st.session_state.data_ready else '#58a6ff'};
       border:1px solid {'#238636' if st.session_state.data_ready else '#1f6feb'}">
    {"✅  COMPLETE" if st.session_state.data_ready else "🔵  STEP 2"}
  </div>
  <div class="step-title">Upload Test Data</div>
  <div class="step-sub">Upload CSV → Select Target & Sensitive Feature</div>
  {"<div class='done-tag'>✓ " + str(len(st.session_state.df_test)) + " rows | Target: " + st.session_state.target_col + " | Sensitive: " + st.session_state.sensitive_col + "</div>" if st.session_state.data_ready else ""}
</div>""", unsafe_allow_html=True)

s3 = "done" if st.session_state.audit_done else ("active" if st.session_state.data_ready else "locked")
c3.markdown(f"""
<div class="step-card {s3}">
  <div class="step-badge" style="background:{'#1a4731' if st.session_state.audit_done else '#3d1f00'};
       color:{'#56d364' if st.session_state.audit_done else '#ffa657'};
       border:1px solid {'#238636' if st.session_state.audit_done else '#d4800a'}">
    {"✅  COMPLETE" if st.session_state.audit_done else "⚡  STEP 3"}
  </div>
  <div class="step-title">Run Fairness Audit</div>
  <div class="step-sub">Fairlearn MetricFrame · DPD · EOD · Mitigation</div>
  {"<div class='done-tag'>✓ Audit complete</div>" if st.session_state.audit_done else ""}
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CONNECT MODEL (Upload PKL OR API)
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📦  STEP 1 — Connect Your Model", expanded=not st.session_state.model_ready):

    st.markdown("### Choose how to connect your model")

    opt1, opt2 = st.columns(2)
    with opt1:
        st.markdown("""
        <div class="opt-card">
          <div class="opt-card-title">📁 Option A — Upload .pkl File</div>
          <div class="opt-card-sub">Upload your trained model saved as a pickle file.<br>Best for: local models, Jupyter-trained models</div>
        </div>
        """, unsafe_allow_html=True)
    with opt2:
        st.markdown("""
        <div class="opt-card">
          <div class="opt-card-title">🔑 Option B — Connect via API</div>
          <div class="opt-card-sub">Client provides API endpoint + API key.<br>Best for: deployed models, client production systems</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    model_option = st.radio(
        "Select connection method",
        ["📁 Upload .pkl file", "🔑 Connect via API endpoint"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # ── OPTION A: UPLOAD PKL ─────────────────────────────────────────
    if model_option == "📁 Upload .pkl file":
        st.markdown("---")
        col_a, col_b = st.columns([1.4, 1])
        with col_a:
            st.markdown("#### Upload your trained model (.pkl)")
            st.markdown("""<div class="ibox">
            <span style="color:#56d364">✓</span>  Works with <b style="color:#e6edf3">ANY sklearn model</b> — any domain<br>
            <span style="color:#56d364">✓</span>  Model must be <b style="color:#e6edf3">already trained</b> — no training here<br>
            <span style="color:#56d364">✓</span>  Saved using <b style="color:#e6edf3">pickle.dump(model, f)</b><br>
            <span style="color:#ffa657">⚠</span>  Must support <b style="color:#e6edf3">.predict()</b> method
            </div>""", unsafe_allow_html=True)

            pkl_file = st.file_uploader(
                "Choose .pkl file",
                type=["pkl"],
                key="pkl_uploader",
                label_visibility="collapsed"
            )
            if pkl_file:
                try:
                    model_obj = pickle.load(pkl_file)
                    model_obj = fix_sklearn_model_compatibility(model_obj)
                    if not hasattr(model_obj, 'predict'):
                        st.error("❌ Not a valid model — must have .predict() method")
                    else:
                        model_name = type(model_obj).__name__
                        st.success(f"✅ Model loaded: **{model_name}** | `{pkl_file.name}`")
                        if st.button("✅  Confirm — Use This Model", key="confirm_pkl"):
                            st.session_state.model_obj   = model_obj
                            st.session_state.model_name  = model_name
                            st.session_state.model_source= "upload"
                            st.session_state.model_ready = True
                            st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed: {e}")

        with col_b:
            st.markdown("#### How to save model as .pkl")
            st.code("""import pickle

with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved!")""", language="python")

    # ── OPTION B: API KEY ────────────────────────────────────────────
    elif model_option == "🔑 Connect via API endpoint":
        st.markdown("---")
        st.markdown("#### Enter Client API Details")

        st.markdown("""<div class="ibox" style="margin-bottom:1rem">
        <span style="color:#58a6ff">ℹ</span>  Client gives you: <b style="color:#e6edf3">API Endpoint URL</b> + <b style="color:#e6edf3">API Key</b><br>
        <span style="color:#56d364">✓</span>  FairAudit sends test data rows to the API → gets predictions back<br>
        <span style="color:#56d364">✓</span>  Fairlearn runs audit on those predictions<br>
        <span style="color:#56d364">✓</span>  No model file needed — works with any deployed model<br>
        <span style="color:#ffa657">⚠</span>  API must accept JSON input and return prediction (0 or 1) per row
        </div>""", unsafe_allow_html=True)

        col_api1, col_api2 = st.columns(2)

        with col_api1:
            api_endpoint = st.text_input(
                "🌐 API Endpoint URL",
                placeholder="https://your-client-api.com/predict",
                help="The URL that accepts POST requests with row data and returns predictions"
            )
            api_key = st.text_input(
                "🔑 API Key",
                placeholder="Enter API key here...",
                type="password",
                help="The authentication key provided by the client"
            )

        with col_api2:
            api_format = st.selectbox(
                "📋 Request Format",
                ["JSON (rows as list)", "JSON (single row)", "REST API (standard)"],
                help="How the API expects data to be sent"
            )
            api_model_name = st.text_input(
                "📝 Model Name (for display)",
                placeholder="e.g. ClientCreditModel_v2",
                help="Just a label to identify this model in the audit report"
            )

        # API Format explanation
        st.markdown("#### How the API will be called")
        if "JSON (rows as list)" in api_format:
            st.code("""{
  "api_key": "your_api_key",
  "data": [
    {"age": 45, "gender": 1, "bmi": 28.3, ...},
    {"age": 62, "gender": 0, "bmi": 31.1, ...}
  ]
}
→ Expected Response: {"predictions": [0, 1, 0, 1, ...]}""", language="json")
        elif "single row" in api_format:
            st.code("""{
  "api_key": "your_api_key",
  "age": 45,
  "gender": 1,
  "bmi": 28.3,
  ...
}
→ Expected Response: {"prediction": 1}""", language="json")
        else:
            st.code("""Headers: {"Authorization": "Bearer your_api_key"}
Body:    {"instances": [[45, 1, 28.3, ...]]}

→ Expected Response: {"predictions": [0, 1, 0, ...]}""", language="json")

        # Test Connection button
        st.markdown("---")
        col_test, col_confirm = st.columns(2)

        with col_test:
            if st.button("🔌 Test API Connection", key="test_api"):
                if not api_endpoint:
                    st.error("❌ Please enter API endpoint URL")
                elif not api_key:
                    st.error("❌ Please enter API key")
                else:
                    with st.spinner("Testing connection..."):
                        try:
                            headers = {"Authorization": f"Bearer {api_key}",
                                       "Content-Type": "application/json"}
                            test_payload = {"api_key": api_key, "test": True}
                            response = requests.post(
                                api_endpoint,
                                headers=headers,
                                json=test_payload,
                                timeout=10
                            )
                            if response.status_code in [200, 201, 422]:
                                st.success(f"✅ API reachable! Status: {response.status_code}")
                            elif response.status_code == 401:
                                st.error("❌ Authentication failed — check your API key")
                            elif response.status_code == 404:
                                st.error("❌ Endpoint not found — check the URL")
                            else:
                                st.warning(f"⚠️ API responded with status: {response.status_code}")
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Cannot reach API — check the endpoint URL")
                        except requests.exceptions.Timeout:
                            st.error("❌ Connection timed out — API may be slow")
                        except Exception as e:
                            st.error(f"❌ Connection error: {e}")

        with col_confirm:
            if st.button("✅  Confirm API — Go to Step 2", key="confirm_api"):
                if not api_endpoint:
                    st.error("❌ Please enter API endpoint URL")
                elif not api_key:
                    st.error("❌ Please enter API key")
                else:
                    name = api_model_name if api_model_name else "API_Model"
                    st.session_state.model_obj    = None
                    st.session_state.model_name   = name
                    st.session_state.model_source = "api"
                    st.session_state.api_endpoint = api_endpoint
                    st.session_state.api_key      = api_key
                    st.session_state.api_format   = api_format
                    st.session_state.model_ready  = True
                    st.success(f"✅ API configured: {name}")
                    st.rerun()

        # How it works info box
        st.markdown("---")
        st.markdown("""<div class="ibox">
        <span style="color:#58a6ff">📌 How API mode works in FairAudit:</span><br><br>
        <span style="color:#56d364">1.</span>  You upload test data CSV in Step 2<br>
        <span style="color:#56d364">2.</span>  FairAudit sends each row to the client's API endpoint<br>
        <span style="color:#56d364">3.</span>  API returns predictions (0 or 1) for each row<br>
        <span style="color:#56d364">4.</span>  Fairlearn runs MetricFrame, DPD, EOD on those predictions<br>
        <span style="color:#56d364">5.</span>  Bias report generated — client never needs to share their model<br><br>
        <span style="color:#ffa657">⚠</span>  This is the real-world enterprise fairness audit workflow
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — UPLOAD TEST DATA
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📂  STEP 2 — Upload Test Data & Configure", expanded=st.session_state.model_ready and not st.session_state.data_ready):
    if not st.session_state.model_ready:
        st.markdown('<div class="ibox"><span style="color:#ffa657">⚠</span>  Complete <b style="color:#e6edf3">Step 1</b> first</div>', unsafe_allow_html=True)
    else:
        # Show which mode is active
        if st.session_state.model_source == "api":
            st.markdown(f"""<div class="ibox" style="margin-bottom:1rem">
            <span style="color:#56d364">✓</span>  API Mode: <b style="color:#e6edf3">{st.session_state.model_name}</b><br>
            <span style="color:#58a6ff">ℹ</span>  Endpoint: <b style="color:#e6edf3">{st.session_state.api_endpoint}</b><br>
            <span style="color:#58a6ff">ℹ</span>  FairAudit will call this API with your test data to get predictions
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="ibox" style="margin-bottom:1rem">
            <span style="color:#56d364">✓</span>  Model loaded: <b style="color:#e6edf3">{st.session_state.model_name}</b> (local file)
            </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns([1.4, 1])
        with col_a:
            st.markdown("#### Upload test CSV file")
            st.markdown("""<div class="ibox">
            <span style="color:#56d364">✓</span>  Any CSV — healthcare, finance, cybersecurity, HR<br>
            <span style="color:#56d364">✓</span>  Must have <b style="color:#e6edf3">true label column</b> (actual outcomes 0/1)<br>
            <span style="color:#56d364">✓</span>  Must have <b style="color:#e6edf3">sensitive feature column</b> (gender, age, race...)
            </div>""", unsafe_allow_html=True)
            csv_file = st.file_uploader("Choose CSV", type=["csv"], key="csv_uploader", label_visibility="collapsed")
            df_loaded = None
            if csv_file:
                df_loaded = pd.read_csv(csv_file)
                st.success(f"✅ `{csv_file.name}` — {len(df_loaded):,} rows, {df_loaded.shape[1]} columns")
                st.markdown("#### 👁 Preview")
                st.dataframe(df_loaded.head(8), use_container_width=True)
        with col_b:
            st.markdown("#### Save test data from Jupyter")
            st.code("""test_df = X_test.copy()
test_df["target"]    = y_test.values
test_df["sensitive"] = sf_test.values
test_df.to_csv("test_data.csv", index=False)
print("✅ Saved!")""", language="python")

        if df_loaded is not None:
            st.markdown("---")
            st.markdown("### ⚙️ Configure Audit")
            col_t, col_s = st.columns(2)
            with col_t:
                all_cols   = list(df_loaded.columns)
                default_t  = all_cols[-1]
                for hint in ["stroke","label","target","attrition","loan_status","default","churn"]:
                    if hint in [c.lower() for c in all_cols]:
                        default_t = [c for c in all_cols if c.lower()==hint][0]
                        break
                target_col = st.selectbox(
                    "🎯 Target column — true labels (0/1)",
                    all_cols,
                    index=all_cols.index(default_t),
                    help="Actual real outcome column"
                )
            with col_s:
                sens_cols     = [c for c in df_loaded.columns if c != target_col]
                default_s     = sens_cols[0]
                for hint in ["gender","sex","age","race","proto","department","ethnicity"]:
                    if hint in [c.lower() for c in sens_cols]:
                        default_s = [c for c in sens_cols if c.lower()==hint][0]
                        break
                sensitive_col = st.selectbox(
                    "⚖️ Sensitive feature — fairness group",
                    sens_cols,
                    index=sens_cols.index(default_s),
                    help="Column to test fairness across"
                )

            if target_col and sensitive_col:
                groups = df_loaded[sensitive_col].unique()
                dist   = df_loaded[target_col].value_counts().to_dict()
                st.markdown(f"""<div class="ibox">
                <span style="color:#58a6ff">ℹ</span>  Model: <b style="color:#e6edf3">{st.session_state.model_name}</b>
                  ({st.session_state.model_source.upper()})<br>
                <span style="color:#58a6ff">ℹ</span>  Target: <b style="color:#e6edf3">{target_col}</b>
                  — values: {dict(list(dist.items())[:4])}<br>
                <span style="color:#58a6ff">ℹ</span>  Sensitive: <b style="color:#e6edf3">{sensitive_col}</b>
                  — groups: {list(groups[:6])}<br>
                <span style="color:#58a6ff">ℹ</span>  Test rows: <b style="color:#e6edf3">{len(df_loaded):,}</b>
                </div>""", unsafe_allow_html=True)

                if st.button("✅  Confirm — Go to Audit", key="confirm_data"):
                    st.session_state.df_test       = df_loaded
                    st.session_state.target_col    = target_col
                    st.session_state.sensitive_col = sensitive_col
                    st.session_state.data_ready    = True
                    st.rerun()


        # ── PREDICTION OUTPUT VIEWER ─────────────────────────
        if st.session_state.data_ready:
            st.markdown("---")
            st.markdown('''<div class="shdr">🔍 Model Prediction Output</div>''',
                       unsafe_allow_html=True)

            if st.button("▶  Generate Predictions on Test Data",
                        key="gen_predictions"):
                try:
                    model   = st.session_state.model_obj
                    df_test = st.session_state.df_test
                    tc      = st.session_state.target_col
                    sc      = st.session_state.sensitive_col

                    X_test  = df_test.drop(columns=[tc])
                    y_test  = df_test[tc]
                    sf_test = df_test[sc]

                    # Get predictions
                    y_pred  = model.predict(X_test)
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)[:,1].round(4)
                    else:
                        y_prob = ["N/A"] * len(y_pred)

                    # Build output table
                    out = pd.DataFrame()
                    out[sc]              = sf_test.values
                    out["Actual"]        = y_test.values
                    out["Predicted"]     = y_pred
                    out["Confidence %"]  = [
                        f"{v*100:.1f}%" if v != "N/A" else "N/A"
                        for v in y_prob
                    ]
                    out["Result"] = out.apply(
                        lambda r:
                        "✅ Correct"     if r["Actual"]==r["Predicted"]
                        else ("❌ False Alarm" if r["Predicted"]==1
                        else  "⚠️ Missed"), axis=1
                    )

                    # Summary cards
                    total   = len(out)
                    correct = (out["Actual"]==out["Predicted"]).sum()
                    tp = int(((y_pred==1)&(y_test==1)).sum())
                    tn = int(((y_pred==0)&(y_test==0)).sum())
                    fp = int(((y_pred==1)&(y_test==0)).sum())
                    fn = int(((y_pred==0)&(y_test==1)).sum())

                    st.markdown("#### Summary")
                    c1,c2,c3,c4,c5 = st.columns(5)
                    c1.markdown(f'''<div class="mcard">
                        <div class="mval">{total:,}</div>
                        <div class="mlbl">Total Rows</div>
                    </div>''', unsafe_allow_html=True)
                    c2.markdown(f'''<div class="mcard">
                        <div class="mval g">{tp:,}</div>
                        <div class="mlbl">✅ True +ve</div>
                    </div>''', unsafe_allow_html=True)
                    c3.markdown(f'''<div class="mcard">
                        <div class="mval g">{tn:,}</div>
                        <div class="mlbl">✅ True -ve</div>
                    </div>''', unsafe_allow_html=True)
                    c4.markdown(f'''<div class="mcard">
                        <div class="mval o">{fp:,}</div>
                        <div class="mlbl">❌ False Alarm</div>
                    </div>''', unsafe_allow_html=True)
                    c5.markdown(f'''<div class="mcard">
                        <div class="mval r">{fn:,}</div>
                        <div class="mlbl">⚠️ Missed</div>
                    </div>''', unsafe_allow_html=True)

                    st.markdown("---")

                    # Filter options
                    st.markdown("#### Filter & View Predictions")
                    f1,f2,f3 = st.columns(3)
                    with f1:
                        filter_r = st.selectbox(
                            "Filter by Result",
                            ["All","✅ Correct",
                             "❌ False Alarm","⚠️ Missed"],
                            key="pred_filter_r"
                        )
                    with f2:
                        groups = ["All"] + [
                            str(g) for g in
                            sorted(out[sc].unique())
                        ]
                        filter_g = st.selectbox(
                            f"Filter by {sc}",
                            groups,
                            key="pred_filter_g"
                        )
                    with f3:
                        n_rows = st.slider(
                            "Rows to display",
                            10, 500, 50,
                            key="pred_rows"
                        )

                    # Apply filters
                    filtered = out.copy()
                    if filter_r != "All":
                        filtered = filtered[
                            filtered["Result"]==filter_r
                        ]
                    if filter_g != "All":
                        filtered = filtered[
                            filtered[sc].astype(str)==filter_g
                        ]

                    st.markdown(
                        f"Showing **{min(n_rows,len(filtered)):,}**"
                        f" of **{len(filtered):,}** rows"
                    )

                    # Display table
                    st.dataframe(
                        filtered[[sc,"Actual","Predicted",
                                  "Confidence %","Result"
                                ]].head(n_rows)
                                  .reset_index(drop=True),
                        use_container_width=True
                    )

                    # Download button
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️  Download All Predictions as CSV",
                        data=csv,
                        file_name=f"{st.session_state.model_name}_predictions.csv",
                        mime="text/csv",
                        key="dl_pred"
                    )

                    st.success(
                        f"✅ Predictions generated for "
                        f"{total:,} rows — "
                        f"Accuracy: {correct/total:.1%}"
                    )

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

    n1,_,_ = st.columns([1,3,1])
    with n1:
        if st.session_state.model_ready:
            if st.button("← Back to Step 1", key="back1"):
                st.session_state.model_ready = False
                st.session_state.data_ready  = False
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — RUN FAIRNESS AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("▶  STEP 3 — Run Fairness Audit", expanded=st.session_state.data_ready):
    if not st.session_state.data_ready:
        st.markdown('<div class="ibox"><span style="color:#ffa657">⚠</span>  Complete Steps 1 & 2 first</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="ibox" style="margin-bottom:1.5rem">
        <span style="color:#56d364">✓</span>  Model: <b style="color:#e6edf3">{st.session_state.model_name}</b>
          ({st.session_state.model_source.upper()}) &nbsp;|&nbsp;
        <span style="color:#56d364">✓</span>  Rows: <b style="color:#e6edf3">{len(st.session_state.df_test):,}</b> &nbsp;|&nbsp;
        <span style="color:#56d364">✓</span>  Target: <b style="color:#e6edf3">{st.session_state.target_col}</b> &nbsp;|&nbsp;
        <span style="color:#56d364">✓</span>  Sensitive: <b style="color:#58a6ff">{st.session_state.sensitive_col}</b>
        </div>""", unsafe_allow_html=True)

        run_mitigation = st.toggle("Also run bias mitigation", value=True)

        if st.button("▶  Run Fairness Audit Now", key="run_audit"):
            try:
                from fairlearn.metrics import (MetricFrame,
                    demographic_parity_difference,
                    equalized_odds_difference,
                    selection_rate)
                from sklearn.metrics import accuracy_score, precision_score, recall_score

                df_test      = st.session_state.df_test
                tc           = st.session_state.target_col
                sc           = st.session_state.sensitive_col
                X_test       = df_test.drop(columns=[tc])
                y_test       = df_test[tc]
                sf_test      = df_test[sc]

                # ── GET PREDICTIONS (PKL or API) ──────────────────────
                if st.session_state.model_source == "upload":
                    with st.spinner("Running model predictions (local model)..."):
                        y_pred = st.session_state.model_obj.predict(X_test)
                        st.markdown("""<div class="ibox" style="margin-bottom:1rem">
                        <span style="color:#56d364">✓</span>  Predictions generated using <b style="color:#e6edf3">local .pkl model</b>
                        </div>""", unsafe_allow_html=True)

                elif st.session_state.model_source == "api":
                    with st.spinner(f"Calling API: {st.session_state.api_endpoint}..."):
                        try:
                            headers = {
                                "Authorization": f"Bearer {st.session_state.api_key}",
                                "Content-Type":  "application/json",
                                "X-API-Key":     st.session_state.api_key
                            }
                            # Send all rows to API
                            rows = X_test.to_dict(orient="records")
                            payload = {
                                "api_key": st.session_state.api_key,
                                "data":    rows
                            }
                            response = requests.post(
                                st.session_state.api_endpoint,
                                headers=headers,
                                json=payload,
                                timeout=30
                            )
                            if response.status_code == 200:
                                resp_data = response.json()
                                # Try to extract predictions from common response formats
                                if "predictions" in resp_data:
                                    y_pred = np.array(resp_data["predictions"])
                                elif "prediction" in resp_data:
                                    y_pred = np.array(resp_data["prediction"])
                                elif "results" in resp_data:
                                    y_pred = np.array(resp_data["results"])
                                elif isinstance(resp_data, list):
                                    y_pred = np.array(resp_data)
                                else:
                                    st.error(f"❌ Cannot parse API response: {str(resp_data)[:200]}")
                                    st.stop()
                                st.markdown(f"""<div class="ibox" style="margin-bottom:1rem">
                                <span style="color:#56d364">✓</span>  {len(y_pred)} predictions received from API<br>
                                <span style="color:#56d364">✓</span>  Endpoint: <b style="color:#e6edf3">{st.session_state.api_endpoint}</b>
                                </div>""", unsafe_allow_html=True)
                            else:
                                st.error(f"❌ API error {response.status_code}: {response.text[:300]}")
                                st.stop()
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Cannot reach API endpoint — check URL and network")
                            st.stop()
                        except Exception as e:
                            st.error(f"❌ API call failed: {e}")
                            st.stop()

                # ── FAIRLEARN AUDIT ───────────────────────────────────
                with st.spinner("Running Fairlearn audit..."):
                    md = {
                        "accuracy":       accuracy_score,
                        "precision":      lambda yt,yp: precision_score(yt,yp,zero_division=0),
                        "recall":         lambda yt,yp: recall_score(yt,yp,zero_division=0),
                        "selection_rate": selection_rate,
                    }
                    mf  = MetricFrame(metrics=md, y_true=y_test, y_pred=y_pred, sensitive_features=sf_test)
                    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_test)
                    eod = equalized_odds_difference(y_test,    y_pred, sensitive_features=sf_test)

                # ── METRICS ───────────────────────────────────────────
                st.markdown('<div class="shdr">📊 Model Performance on Test Data</div>', unsafe_allow_html=True)
                ov = mf.overall
                def cs(v): return "g" if v<0.05 else ("o" if v<0.10 else "r")
                cols = st.columns(6)
                for col,(val,lbl,cl) in zip(cols,[
                    (f'{ov["accuracy"]:.1%}',      "Accuracy",      "g" if ov["accuracy"]>0.75  else "o"),
                    (f'{ov["precision"]:.1%}',      "Precision",     "g" if ov["precision"]>0.15 else "o"),
                    (f'{ov["recall"]:.1%}',         "Recall",        "g" if ov["recall"]>0.5     else "o"),
                    (f'{ov["selection_rate"]:.1%}', "Selection Rate",""),
                    (f'{abs(dpd):.4f}',             "DPD ↓ better",  cs(abs(dpd))),
                    (f'{abs(eod):.4f}',             "EOD ↓ better",  cs(abs(eod))),
                ]):
                    col.markdown(f'<div class="mcard"><div class="mval {cl}">{val}</div><div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

                # ── VERDICT ───────────────────────────────────────────
                fair = abs(dpd)<0.05 and abs(eod)<0.05
                mild = abs(dpd)<0.10 and abs(eod)<0.10
                if fair:
                    st.markdown('<div class="v-fair">✅  FAIR — DPD & EOD within acceptable threshold (&lt; 0.05)</div>', unsafe_allow_html=True)
                elif mild:
                    st.markdown('<div class="v-mild">⚠️  MILD BIAS — Disparity detected. Mitigation recommended.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="v-bias">🚨  SIGNIFICANT BIAS — Immediate mitigation required.</div>', unsafe_allow_html=True)

                # ── BY GROUP ──────────────────────────────────────────
                st.markdown(f'<div class="shdr">📈 Fairness by {sc} Group</div>', unsafe_allow_html=True)
                ct, cc = st.columns([1, 1.8])
                with ct:
                    st.markdown("**Group breakdown table**")
                    st.dataframe(mf.by_group.style.format("{:.3f}").background_gradient(cmap="RdYlGn",axis=0), use_container_width=True)
                    st.markdown(f"""<div class="ibox" style="margin-top:0.8rem">
                    <span style="color:#58a6ff">DPD</span> {abs(dpd):.4f} — {"✅ Fair" if abs(dpd)<0.05 else "⚠️ Bias"}<br>
                    <span style="color:#58a6ff">EOD</span> {abs(eod):.4f} — {"✅ Fair" if abs(eod)<0.05 else "⚠️ Bias"}<br>
                    <span style="color:#8b949e">Fair threshold: &lt; 0.05</span>
                    </div>""", unsafe_allow_html=True)
                with cc:
                    fig,axes = plt.subplots(1,4,figsize=(13,3.5))
                    fig.patch.set_facecolor("#161b22")
                    for ax,(metric,clr) in zip(axes,[("accuracy","#58a6ff"),("precision","#ffa657"),("recall","#56d364"),("selection_rate","#f85149")]):
                        groups = mf.by_group.index.astype(str)
                        vals   = mf.by_group[metric].values
                        bars   = ax.bar(groups, vals, color=clr, alpha=0.85, width=0.5)
                        ax.set_facecolor("#0d1117")
                        ax.set_title(metric, color="#8b949e", fontsize=9, pad=6)
                        ax.tick_params(colors="#8b949e", labelsize=8)
                        ax.set_xlabel(sc, color="#8b949e", fontsize=8)
                        for spine in ax.spines.values(): spine.set_visible(False)
                        for bar,val in zip(bars,vals):
                            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#8b949e")
                    fig.suptitle(f"{st.session_state.model_name} — Fairness by {sc}", color="#e6edf3", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                # ── MITIGATION ────────────────────────────────────────
                if run_mitigation:
                    st.markdown('<div class="shdr">🔧 Bias Mitigation — ExponentiatedGradient</div>', unsafe_allow_html=True)
                    with st.spinner("Running mitigation..."):
                        try:
                            from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds, EqualizedOdds
                            from sklearn.ensemble import GradientBoostingClassifier
                            from sklearn.model_selection import train_test_split

                            Xm=df_test.drop(columns=[tc]); ym=df_test[tc]; sfm=df_test[sc]
                            Xtr,Xte,ytr,yte,sftr,sfte = train_test_split(Xm,ym,sfm,test_size=0.2,random_state=42)

                            mit = ExponentiatedGradient(
                                estimator=GradientBoostingClassifier(n_estimators=50,random_state=42),
                                constraints=EqualizedOdds()
                            )
                            mit.fit(Xtr, ytr, sensitive_features=sftr)
                            ypm      = mit.predict(Xte)
                            dpd_post = demographic_parity_difference(yte,ypm,sensitive_features=sfte)
                            eod_post = equalized_odds_difference(yte,    ypm,sensitive_features=sfte)
                            dr = (1-abs(dpd_post)/abs(dpd))*100 if abs(dpd)>0 else 0
                            er = (1-abs(eod_post)/abs(eod))*100 if abs(eod)>0 else 0

                            m1,m2,m3,m4 = st.columns(4)
                            m1.markdown(f'<div class="mcard"><div class="mval o">{abs(dpd):.4f}</div><div class="mlbl">DPD Before</div></div>', unsafe_allow_html=True)
                            m2.markdown(f'<div class="mcard"><div class="mval g">{abs(dpd_post):.4f}</div><div class="mlbl">DPD After</div></div>', unsafe_allow_html=True)
                            m3.markdown(f'<div class="mcard"><div class="mval o">{abs(eod):.4f}</div><div class="mlbl">EOD Before</div></div>', unsafe_allow_html=True)
                            m4.markdown(f'<div class="mcard"><div class="mval g">{abs(eod_post):.4f}</div><div class="mlbl">EOD After</div></div>', unsafe_allow_html=True)

                            st.markdown(f"""<div class="ibox" style="margin-top:1rem">
                            <span style="color:#56d364">✓</span>  DPD reduced by <b style="color:#56d364">{dr:.1f}%</b>
                              ({abs(dpd):.4f} → {abs(dpd_post):.4f})<br>
                            <span style="color:#56d364">✓</span>  EOD reduced by <b style="color:#56d364">{er:.1f}%</b>
                              ({abs(eod):.4f} → {abs(eod_post):.4f})<br>
                            <span style="color:#58a6ff">ℹ</span>  Algorithm: ExponentiatedGradient + DemographicParity constraint
                            </div>""", unsafe_allow_html=True)

                            fig2,ax2 = plt.subplots(figsize=(6,3.5))
                            fig2.patch.set_facecolor("#161b22"); ax2.set_facecolor("#0d1117")
                            x=np.arange(2); w=0.3
                            ax2.bar(x-w/2,[abs(dpd),abs(eod)],           w,label="Before",color="#ffa657",alpha=0.85)
                            ax2.bar(x+w/2,[abs(dpd_post),abs(eod_post)], w,label="After", color="#56d364",alpha=0.85)
                            ax2.set_xticks(x); ax2.set_xticklabels(["DPD","EOD"],color="#8b949e",fontsize=12)
                            ax2.tick_params(colors="#8b949e")
                            ax2.axhline(0.05,color="#f85149",linestyle="--",alpha=0.7,linewidth=1.5)
                            ax2.text(1.72,0.052,"Fair threshold (0.05)",color="#f85149",fontsize=9)
                            ax2.legend(facecolor="#161b22",labelcolor="#e6edf3",fontsize=9)
                            ax2.set_title("Before vs After Mitigation",color="#e6edf3",fontsize=11)
                            for sp in ax2.spines.values(): sp.set_visible(False)
                            plt.tight_layout()
                            cl,_ = st.columns([1,1])
                            with cl: st.pyplot(fig2,use_container_width=True)
                            plt.close()

                        except Exception as e:
                            st.warning(f"Mitigation note: {e}")

                st.session_state.audit_done = True
                st.success("🎉  Full audit complete!")

            except Exception as e:
                st.error(f"❌ Audit failed: {e}")

        st.markdown("---")
        if st.button("🔄  Start New Audit", key="reset"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1rem;border-top:1px solid #21262d;text-align:center;
     font-family:Space Mono,monospace;font-size:11px;color:#8b949e;letter-spacing:2px">
  AI MODEL FAIRNESS AUDIT PLATFORM · MICROSOFT FAIRLEARN · UPLOAD OR API · OWASP ML TOP 10 · NIST AI RMF
</div>
""", unsafe_allow_html=True)
