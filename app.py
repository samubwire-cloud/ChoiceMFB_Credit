import streamlit as st
import pandas as pd
import numpy as np
import warnings
import io
import os
import base64
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Choice MFB Credit Intelligence",
    page_icon="https://raw.githubusercontent.com/samubwire-cloud/choice-mfb-credit/main/choice_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── BRAND ─────────────────────────────────────────────────────────────────────
NAVY  = "#110837"
RED   = "#DA2A2F"
LIGHT = "#F5F6FA"

# ── LOGO ──────────────────────────────────────────────────────────────────────
def get_logo():
    paths = [
        "choice_logo.png",
        "/mount/src/choice-mfb-credit/choice_logo.png",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return ""

LOGO = get_logo()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {background:#110837 !important;}
[data-testid="stSidebar"] * {color:white !important;}
[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,0.2)!important;}
.stButton>button{background:#110837!important;color:white!important;
  border:none!important;border-radius:8px!important;font-weight:600!important;}
div[data-testid="stFormSubmitButton"]>button{background:#DA2A2F!important;
  color:white!important;font-weight:700!important;border:none!important;width:100%!important;}
.dec-approve{background:#DCFCE7;color:#14532D;border:1.5px solid #16A34A;
  border-radius:10px;padding:14px 20px;font-size:16px;font-weight:700;margin:12px 0;}
.dec-refer{background:#FEF9C3;color:#713F12;border:1.5px solid #CA8A04;
  border-radius:10px;padding:14px 20px;font-size:16px;font-weight:700;margin:12px 0;}
.dec-decline{background:#FEE2E2;color:#7F1D1D;border:1.5px solid #DC2626;
  border-radius:10px;padding:14px 20px;font-size:16px;font-weight:700;margin:12px 0;}
</style>
""", unsafe_allow_html=True)

# ── MOBILE / iPHONE ICON — JavaScript injection ───────────────────────────────
# Streamlit strips <head> tags from markdown, so we use JS to inject the icon
ICON_URL = "https://raw.githubusercontent.com/samubwire-cloud/choice-mfb-credit/main/choice_favicon.png"

st.markdown(f"""
<script>
(function() {{
    // Remove any existing apple-touch-icon links
    var existing = document.querySelectorAll('link[rel*="apple-touch-icon"]');
    existing.forEach(function(el) {{ el.remove(); }});

    // Inject apple-touch-icon for iPhone home screen
    var link1 = document.createElement('link');
    link1.rel = 'apple-touch-icon';
    link1.sizes = '180x180';
    link1.href = '{ICON_URL}';
    document.head.appendChild(link1);

    // Precomposed version (no shine effect)
    var link2 = document.createElement('link');
    link2.rel = 'apple-touch-icon-precomposed';
    link2.href = '{ICON_URL}';
    document.head.appendChild(link2);

    // Standard favicon
    var link3 = document.createElement('link');
    link3.rel = 'shortcut icon';
    link3.href = '{ICON_URL}';
    document.head.appendChild(link3);

    // PWA theme colour (status bar on iPhone)
    var meta1 = document.createElement('meta');
    meta1.name = 'theme-color';
    meta1.content = '#110837';
    document.head.appendChild(meta1);

    // Make it behave like a full-screen app on iPhone
    var meta2 = document.createElement('meta');
    meta2.name = 'apple-mobile-web-app-capable';
    meta2.content = 'yes';
    document.head.appendChild(meta2);

    var meta3 = document.createElement('meta');
    meta3.name = 'apple-mobile-web-app-title';
    meta3.content = 'Choice MFB';
    document.head.appendChild(meta3);

    var meta4 = document.createElement('meta');
    meta4.name = 'apple-mobile-web-app-status-bar-style';
    meta4.content = 'black-translucent';
    document.head.appendChild(meta4);

    console.log('Choice MFB: iPhone icons injected');
}})();
</script>
""", unsafe_allow_html=True)

# ── BANK CONFIG ───────────────────────────────────────────────────────────────
COF    = 0.12
OPEX   = 0.04
PROFIT = 0.03
CAPRAT = 0.08
LGD_D  = 0.60

PRICING = [
    ("AAA", 750, 0.23), ("AA",  700, 0.25), ("A",   650, 0.28),
    ("BBB", 620, 0.33), ("BB",  580, 0.40), ("B",   500, 0.55),
    ("C",     0, 0.72),
]

PRODUCT_LGD = {
    "Chemsha Biashara BC Logbook Loan": 0.35,
    "Chemsha Biashara BC Asset Finance": 0.40,
    "Staff Development Loan": 0.40,
    "Salary Advance": 0.40,
    "Choice Unsecured Loan": 0.75,
    "Boya Discounted Working Capital": 0.60,
}

BASE = 600
PDO  = 20
FAC  = PDO / np.log(2)
OFF  = BASE

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        sc  = joblib.load("scorecard_model.pkl")
        ews = joblib.load("ews_model.pkl")
        return sc, ews, True
    except Exception:
        np.random.seed(42)
        n = 400
        cols = ["w_ba","w_bt","w_br","w_bg","w_bs"]
        X = pd.DataFrame({c: np.random.normal(0, 0.5, n) for c in cols})
        y = ((X["w_bg"] + X["w_ba"] + np.random.normal(0, 0.3, n)) > 0.3).astype(int)
        lr = LogisticRegression(random_state=42, max_iter=500)
        lr.fit(X, y)
        sc = {
            "model": lr,
            "woe_maps": {c: {} for c in ["ba","bt","br","bg","bs"]},
            "bin_cols": ["ba","bt","br","bg","bs"],
            "woe_feats": cols,
        }
        Xe = pd.DataFrame({
            "sig_days_since_payment": np.abs(np.random.normal(20, 40, n)),
            "sig_balance_erosion": np.random.uniform(0.3, 1.2, n),
            "sig_loan_age": np.abs(np.random.normal(10, 12, n)),
            "sig_ltv": np.random.uniform(0.3, 2.5, n),
        })
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced")
        rf.fit(Xe, y)
        ews = {"model": rf, "features": list(Xe.columns)}
        return sc, ews, False

sc_bundle, ews_bundle, prod_mode = load_models()
sc_model  = sc_bundle["model"]
woe_maps  = sc_bundle["woe_maps"]
BIN_COLS  = sc_bundle["bin_cols"]
ews_model = ews_bundle["model"]

# ── ENGINE ────────────────────────────────────────────────────────────────────
def bv(v, t):
    if t == "amount":
        if pd.isna(v) or v <= 100000:  return "A"
        elif v <= 300000:               return "B"
        elif v <= 500000:               return "C"
        elif v <= 1000000:              return "D"
        elif v <= 2000000:              return "E"
        else:                           return "F"
    elif t == "term":
        if pd.isna(v) or v <= 3:  return "A"
        elif v <= 6:               return "B"
        elif v <= 12:              return "C"
        elif v <= 18:              return "D"
        elif v <= 24:              return "E"
        else:                      return "F"
    elif t == "rate":
        if pd.isna(v) or v == 0:  return "A"
        elif v <= 24:              return "B"
        elif v <= 36:              return "C"
        elif v <= 48:              return "D"
        else:                      return "E"
    elif t == "age":
        if pd.isna(v):   return "Unknown"
        elif v <= 6:     return "A"
        elif v <= 12:    return "B"
        elif v <= 24:    return "C"
        elif v <= 48:    return "D"
        else:            return "E"
    elif t == "sector":
        if pd.isna(v): return "C"
        s = str(v)
        if any(h in s for h in ["Communication Network", "Transport infrastructure"]):
            return "B"
        if any(l in s for l in ["Education", "Professional Bodies",
                                  "Home Development", "Home Improvements", "Real Estate"]):
            return "A"
        return "C"

def get_grade(score):
    for name, min_sc, rate in PRICING:
        if score >= min_sc:
            return name, rate
    return "C", 0.72

def score_app(la, rt, ir=0, age=0, sec="Wholesale and Retail"):
    bins = {
        "ba": bv(la, "amount"), "bt": bv(rt, "term"),
        "br": bv(ir or 0, "rate"), "bg": bv(age, "age"), "bs": bv(sec, "sector"),
    }
    woes = [woe_maps[c].get(str(bins[c]), 0.0) for c in BIN_COLS]
    X = np.array(woes).reshape(1, -1)
    pd_v = float(sc_model.predict_proba(X)[0, 1])
    p = np.clip(pd_v, 0.0001, 0.9999)
    sc = int(np.clip(OFF + FAC * np.log((1 - p) / p), 300, 850))
    grade, _ = get_grade(sc)
    return {"pd_value": round(pd_v, 4), "score": sc, "grade": grade}

def price_l(pd_v, la, col=0, prod="Standard"):
    lgd = PRODUCT_LGD.get(prod, LGD_D)
    el_r = pd_v * lgd
    el_k = el_r * la
    rbp  = float(np.clip(COF + OPEX + el_r + PROFIT, 0.19, 0.72))
    sc2  = int(np.clip(OFF + FAC * np.log(
        (1 - np.clip(pd_v, 0.0001, 0.9999)) / np.clip(pd_v, 0.0001, 0.9999)
    ), 300, 850))
    _, gr_r = get_grade(sc2)
    rec = round(max(rbp, gr_r), 4)
    raroc = (la * (rec - COF - OPEX) - el_k) / (la * CAPRAT) if la > 0 else 0
    return {
        "lgd": lgd, "el_rate": round(el_r, 4), "el_kes": round(el_k),
        "rbp_floor": round(rbp, 4), "recommended_rate": rec, "raroc": round(raroc, 4),
    }

def ews_check(dsp=5, be=0.5, lam=0, ltv=0.5):
    feats = np.array([[
        float(np.clip(dsp, 0, 2800)),
        float(np.clip(be,  0, 2)),
        float(np.clip(lam, 0, 120)),
        float(np.clip(ltv, 0, 5)),
    ]])
    prob = float(ews_model.predict_proba(feats)[0, 1])
    sc   = round(prob * 100, 1)
    if   sc >= 70: flag = "RED";    act = "Immediate escalation — same day"
    elif sc >= 45: flag = "AMBER";  act = "Proactive contact within 48 hours"
    elif sc >= 25: flag = "YELLOW"; act = "Schedule check-in within 5 days"
    else:          flag = "GREEN";  act = "No action — monthly monitoring"
    trg = []
    if dsp >= 45:              trg.append("No payment for 45+ days")
    if 30 <= dsp < 45:         trg.append("No payment for 30+ days")
    if be >= 0.95 and lam > 6: trg.append("Low repayment velocity")
    if lam >= 18:              trg.append(f"Loan age {lam:.0f} months — maturity risk")
    if ltv > 1.0:              trg.append(f"LTV breach ({ltv:.0%})")
    return {"ews_score": sc, "ews_flag": flag, "ews_action": act, "triggers": trg}

def full_appraisal(name, la, rt, sec, prod, col=0, age=0, dsp=0, be=1.0):
    s = score_app(la, rt, 0, age, sec)
    p = price_l(s["pd_value"], la, col, prod)
    e = ews_check(dsp, be, age, col / la if col > 0 else 2.0)
    if   s["score"] >= 650: dec = "✅ AUTO-APPROVE"; dc = "dec-approve"
    elif s["score"] >= 580: dec = "✅ APPROVE";      dc = "dec-approve"
    elif s["score"] >= 500: dec = "⟳ REFER TO COMMITTEE"; dc = "dec-refer"
    else:                   dec = "❌ DECLINE";      dc = "dec-decline"
    if e["ews_flag"] == "RED" and s["score"] >= 580:
        dec = "⟳ REFER — EWS RED FLAG"; dc = "dec-refer"
    rm = p["recommended_rate"] / 12
    mo = la * rm * (1 + rm)**rt / ((1 + rm)**rt - 1) if rm > 0 and rt > 0 else la / rt
    return {**s, **p, **e, "decision": dec, "dec_class": dc,
            "monthly_payment": round(mo), "applicant_name": name,
            "loan_amount": la, "repayment_term": rt, "collateral_amount": col}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO:
        st.markdown(
            '<div style="background:white;border-radius:8px;padding:10px;'
            'margin-bottom:16px;text-align:center">'
            f'<img src="data:image/png;base64,{LOGO}" style="width:100%;max-width:200px"/>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="color:white;font-size:18px;font-weight:700;margin-bottom:16px">'
            '🏦 Choice MFB</div>',
            unsafe_allow_html=True
        )

    page = st.radio("", [
        "🏠  Dashboard",
        "📋  Loan Appraisal",
        "📊  Portfolio Analytics",
        "🚦  EWS Monitor",
        "📁  Batch Processing",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(f"Models: {'✅ Production' if prod_mode else '⚠️ Demo'}")
    st.caption("Riverside Branch · May 2026")

# ── HEADER ────────────────────────────────────────────────────────────────────
logo_html = (
    f'<img src="data:image/png;base64,{LOGO}" style="height:38px"/>'
    if LOGO else
    '<span style="color:white;font-size:18px;font-weight:700">🏦 Choice MFB</span>'
)
st.markdown(
    f'<div style="background:{NAVY};padding:14px 20px;border-radius:10px;'
    f'display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">'
    f'{logo_html}'
    f'<div style="text-align:right">'
    f'<div style="color:white;font-size:15px;font-weight:700">Credit Intelligence Platform</div>'
    f'<div style="color:rgba(255,255,255,0.6);font-size:11px">Samuel · Head of Credit · Riverside Branch</div>'
    f'</div></div>',
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown("### Portfolio Overview — May 2026")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Accounts", "725")
    c2.metric("Outstanding",    "KES 746M")
    c3.metric("NPL Ratio",      "9.0%",  delta="+4.0% vs benchmark", delta_color="inverse")
    c4.metric("PAR 30",         "9.0%",  delta="+4.0% vs benchmark", delta_color="inverse")
    c5.metric("Prov. Coverage", "91%")
    c6.metric("Products",       "23")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PAR Aging (KES Millions)")
        par_df = pd.DataFrame({
            "Bucket": ["Current","PAR 1-30d","PAR 31-60d","PAR 61-90d",
                       "PAR 91-180d","PAR 181-365d","PAR >365d"],
            "KES_M":  [224.5, 81.2, 20.5, 8.1, 18.5, 11.0, 9.4],
        })
        fig = go.Figure(go.Bar(
            x=par_df["Bucket"], y=par_df["KES_M"],
            marker_color=["#16A34A","#CA8A04","#EA580C","#DC2626","#991B1B","#7F1D1D","#450A0A"],
            text=[f"{v:.1f}M" for v in par_df["KES_M"]],
            textposition="outside",
        ))
        fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                          yaxis=dict(gridcolor="#F3F4F6"), showlegend=False,
                          margin=dict(t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Loan Classification")
        clf_df = pd.DataFrame({
            "Class": ["Normal","Watch","Substandard","Doubtful","Loss"],
            "Value": [597.8, 81.2, 20.5, 8.1, 38.9],
        })
        fig2 = go.Figure(go.Pie(
            labels=clf_df["Class"], values=clf_df["Value"], hole=0.45,
            marker_colors=["#16A34A","#CA8A04","#EA580C","#DC2626","#7F1D1D"],
            textfont_size=11,
        ))
        fig2.update_layout(
            height=300, paper_bgcolor="white", showlegend=True,
            annotations=[dict(text="KES 746M", x=0.5, y=0.5, font_size=12, showarrow=False)],
            margin=dict(t=10,b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    ca, cb, cc = st.columns(3)
    with ca:
        st.subheader("EWS Status")
        st.error("🔴 RED — 168 accounts · KES 60.6M")
        st.warning("🟠 AMBER — 13 accounts · KES 1.1M")
        st.warning("🟡 YELLOW — 23 accounts")
        st.success("🟢 GREEN — 521 accounts")
    with cb:
        st.subheader("Pricing Health")
        st.metric("Underpriced loans",    "174", help="RBP > Actual rate")
        st.metric("Negative RAROC loans", "218", help="Destroying shareholder value")
        st.metric("RBP income uplift",    "KES 42.2M pa")
    with cc:
        st.subheader("Top Actions")
        st.markdown("""
- 🔴 **Call 13 AMBER** accounts today
- 🔴 **Write off** Platinum Imports (2,719 days)
- 🟡 **Reprice** Staff Dev Loans (8.4% rate)
- 🟡 **Review** 38 underwater loans (LTV > 100%)
- 🟢 May-26 collections exceeded disbursals ✓
        """)

# ══════════════════════════════════════════════════════════════════════════════
# LOAN APPRAISAL
# ══════════════════════════════════════════════════════════════════════════════
elif "Appraisal" in page:
    st.markdown("### New Loan Appraisal")

    with st.form("appraisal_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name    = st.text_input("Client Name", placeholder="e.g. WANJIKU TRADERS LTD")
            product = st.selectbox("Product", list(PRODUCT_LGD.keys()) + ["Other"])
            sector  = st.selectbox("Sector", [
                "Wholesale and Retail",
                "Transport infrastructure development services",
                "Home Development", "Professional Bodies", "Education",
                "Communication Network Systems", "Real Estate Development", "Other",
            ])
        with c2:
            la  = st.number_input("Loan Amount (KES)",  10_000, 5_000_000, 500_000, 10_000, format="%d")
            rt  = st.selectbox("Term (months)", [3,6,9,12,18,24,36,48,60], index=3)
            col = st.number_input("Collateral (KES)",   0, 20_000_000, 0, 50_000, format="%d")
        with c3:
            age = st.number_input("Loan Age (months) — 0 for new", 0, 120, 0)
            dsp = st.number_input("Days Since Last Payment — 0 for new", 0, 2800, 0)
            be  = st.slider("Balance Erosion — 1.0 for new loan", 0.1, 2.0, 1.0, 0.05)

        submitted = st.form_submit_button("🔍 RUN CREDIT APPRAISAL", use_container_width=True)

    if submitted:
        if not name:
            st.error("Please enter a client name.")
        else:
            with st.spinner("Running appraisal..."):
                r = full_appraisal(name, la, rt, sector, product, col, age, dsp, be)

            st.markdown(f'<div class="{r["dec_class"]}">{r["decision"]}</div>',
                        unsafe_allow_html=True)

            k1,k2,k3,k4,k5,k6 = st.columns(6)
            k1.metric("Score",   f"{r['score']}/850")
            k2.metric("Grade",   r["grade"])
            k3.metric("PD",      f"{r['pd_value']:.1%}")
            k4.metric("Rate",    f"{r['recommended_rate']:.1%}")
            k5.metric("RAROC",   f"{r['raroc']:.1%}")
            k6.metric("Monthly", f"KES {r['monthly_payment']:,.0f}")

            ca, cb = st.columns(2)
            with ca:
                st.subheader("Risk-Based Pricing")
                df_p = pd.DataFrame([
                    {"Component": "Cost of Funds",       "Rate": f"{COF:.1%}"},
                    {"Component": "Operating Expenses",  "Rate": f"{OPEX:.1%}"},
                    {"Component": f"Expected Loss (PD×LGD {r['lgd']:.0%})", "Rate": f"{r['el_rate']:.1%}"},
                    {"Component": "Target Profit",       "Rate": f"{PROFIT:.1%}"},
                    {"Component": "RBP Floor",           "Rate": f"{r['rbp_floor']:.1%}"},
                    {"Component": "✅ RECOMMENDED RATE", "Rate": f"{r['recommended_rate']:.1%}"},
                ])
                st.dataframe(df_p, hide_index=True, use_container_width=True)
                ltv = col / la if col > 0 else 0
                st.caption(f"LTV: {ltv:.0%} | Annual EL: KES {r['el_kes']:,.0f} | RAROC: {r['raroc']:.1%}")

            with cb:
                st.subheader("Early Warning Assessment")
                emoji_map = {"RED":"🔴","AMBER":"🟠","YELLOW":"🟡","GREEN":"🟢"}
                em = emoji_map.get(r["ews_flag"], "🟢")
                msg = f"{em} {r['ews_flag']} — Score {r['ews_score']:.0f}/100"
                if   r["ews_flag"] == "RED":    st.error(msg)
                elif r["ews_flag"] in ("AMBER","YELLOW"): st.warning(msg)
                else:                           st.success(msg)
                st.write(f"**Action:** {r['ews_action']}")
                for t in r["triggers"]:
                    st.warning(f"⚑ {t}")
                if not r["triggers"]:
                    st.success("✓ No warning triggers. Loan appears healthy.")

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=r["score"],
                title={"text": f"{name} — Grade {r['grade']}", "font": {"size": 13}},
                gauge={
                    "axis": {"range": [300, 850]}, "bar": {"color": NAVY},
                    "steps": [
                        {"range": [300, 500], "color": "#FEE2E2"},
                        {"range": [500, 580], "color": "#FFEDD5"},
                        {"range": [580, 650], "color": "#FEF9C3"},
                        {"range": [650, 850], "color": "#DCFCE7"},
                    ],
                }
            ))
            fig_g.update_layout(height=250, margin=dict(t=40,b=0,l=20,r=20),
                                  paper_bgcolor="white")
            st.plotly_chart(fig_g, use_container_width=True)

            memo = (
                f"CHOICE MICROFINANCE BANK LIMITED\n"
                f"CREDIT APPRAISAL MEMO\n"
                f"{'='*50}\n"
                f"Client:    {name}\n"
                f"Product:   {product}\n"
                f"Sector:    {sector}\n"
                f"{'─'*50}\n"
                f"Loan:      KES {la:,.0f}\n"
                f"Term:      {rt} months\n"
                f"Collateral:KES {col:,.0f}  LTV: {ltv:.0%}\n"
                f"{'─'*50}\n"
                f"Score:     {r['score']}/850  Grade: {r['grade']}\n"
                f"PD:        {r['pd_value']:.1%}\n"
                f"EWS:       {r['ews_flag']} ({r['ews_score']:.0f}/100)\n"
                f"{'─'*50}\n"
                f"Rate:      {r['recommended_rate']:.1%}\n"
                f"RAROC:     {r['raroc']:.1%}\n"
                f"Monthly:   KES {r['monthly_payment']:,.0f}\n"
                f"{'─'*50}\n"
                f"DECISION:  {r['decision']}\n"
                f"{'='*50}\n"
            )
            st.download_button(
                "⬇️ Download Credit Memo", memo.encode(),
                f"Memo_{name.replace(' ','_')}.txt",
                use_container_width=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown("### Portfolio Analytics")
    up = st.file_uploader("Upload monthly loan book (.xlsx)", type=["xlsx"])

    @st.cache_data
    def load_book(fb):
        xl = pd.ExcelFile(io.BytesIO(fb))
        sheet = xl.sheet_names[1] if len(xl.sheet_names) > 1 else xl.sheet_names[0]
        df = None
        for hdr in [4, 3, 2, 0]:
            try:
                tmp = pd.read_excel(io.BytesIO(fb), sheet_name=sheet, header=hdr)
                tmp = tmp.iloc[:, :70].dropna(how="all")
                cols_lower = [str(c).lower() for c in tmp.columns]
                if any("balance" in c or "amount" in c or "loan" in c for c in cols_lower):
                    df = tmp
                    break
            except Exception:
                continue
        if df is None:
            df = pd.read_excel(io.BytesIO(fb), sheet_name=sheet, header=4)
            df = df.iloc[:, :70].dropna(how="all")
        df.columns = [str(c).strip() for c in df.columns]
        num_cols = ["Loan Amount","OS Balance","Arrears Days","Arrears Amount",
                    "Interest Rate","Provision","Collateral Amount","Repayment Term",
                    "Principle Balance","Installment Amount"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "OS Balance" not in df.columns:
            for col in df.columns:
                if "balance" in col.lower():
                    df["OS Balance"] = pd.to_numeric(df[col], errors="coerce")
                    break
            if "OS Balance" not in df.columns:
                df["OS Balance"] = 0
        if "Arrears Days" not in df.columns:
            df["Arrears Days"] = 0
        if "Provision" not in df.columns:
            df["Provision"] = 0
        if "Product Name" not in df.columns:
            for col in df.columns:
                if "product" in col.lower():
                    df["Product Name"] = df[col]
                    break
            if "Product Name" not in df.columns:
                df["Product Name"] = "Unknown"
        for dc in ["Disbursed On","Last Repayment Date"]:
            if dc in df.columns:
                df[dc] = pd.to_datetime(df[dc], errors="coerce")
        if "Classification" in df.columns:
            cm = {"Normal":"Normal","Watch":"Watch","Substandard":"Substandard",
                  "doubtful":"Doubtful","loss":"Loss", 0:"Normal"}
            df["Classification"] = df["Classification"].map(cm).fillna("Unknown")
        else:
            df["Classification"] = "Unknown"
        df["bad"] = df["Classification"].isin(["Substandard","Doubtful","Loss"]).astype(int)
        def pb(d):
            if pd.isna(d): return "Unknown"
            elif d == 0:    return "1.Current"
            elif d <= 30:   return "2.PAR 1-30d"
            elif d <= 60:   return "3.PAR 31-60d"
            elif d <= 90:   return "4.PAR 61-90d"
            elif d <= 180:  return "5.PAR 91-180d"
            elif d <= 365:  return "6.PAR 181-365d"
            else:           return "7.PAR >365d"
        df["par_bucket"] = df["Arrears Days"].fillna(0).apply(pb)
        return df

    if up:
        df = load_book(up.read())
        total_os = df["OS Balance"].sum()
        npl_v    = df.loc[df["bad"]==1, "OS Balance"].sum()
        prov     = df["Provision"].sum()

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Accounts",       f"{len(df):,}")
        m2.metric("OS Balance",     f"KES {total_os/1e6:.1f}M")
        m3.metric("NPL Ratio",      f"{npl_v/total_os*100:.1f}%")
        m4.metric("Prov. Coverage", f"{prov/npl_v*100:.0f}%" if npl_v > 0 else "N/A")
        m5.metric("Bad Rate",       f"{df['bad'].mean():.1%}")

        c1, c2 = st.columns(2)
        with c1:
            par_s = df.groupby("par_bucket")["OS Balance"].sum().reset_index()
            fig = px.bar(par_s, x="par_bucket", y="OS Balance", title="PAR Aging",
                         color_discrete_sequence=[NAVY])
            fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            clf_s = df.groupby("Classification")["OS Balance"].sum().reset_index()
            cmap  = {"Normal":"#16A34A","Watch":"#CA8A04","Substandard":"#EA580C",
                     "Doubtful":"#DC2626","Loss":"#7F1D1D","Unknown":"#9CA3AF"}
            fig2 = px.pie(clf_s, values="OS Balance", names="Classification",
                          hole=0.4, title="Classification",
                          color="Classification", color_discrete_map=cmap)
            fig2.update_layout(height=300, paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

        prod_s = df.groupby("Product Name").agg(
            count=("bad","count"), npl_rate=("bad","mean"),
            os=("OS Balance","sum")).reset_index()
        prod_s = prod_s[prod_s["count"] >= 3].sort_values("os", ascending=False)
        prod_s["npl_pct"] = prod_s["npl_rate"] * 100
        fig3 = px.bar(prod_s, x="Product Name", y="npl_pct",
                      title="NPL Rate by Product",
                      color="npl_pct",
                      color_continuous_scale=[(0,"#16A34A"),(0.3,"#CA8A04"),(1,"#DC2626")],
                      text=prod_s["npl_pct"].apply(lambda x: f"{x:.0f}%"),
                      labels={"npl_pct":"NPL Rate (%)","Product Name":""})
        fig3.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                            xaxis_tickangle=-30, margin=dict(t=30,b=80))
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("📂 Upload your BSM loan book Excel file to see live portfolio analytics.")

# ══════════════════════════════════════════════════════════════════════════════
# EWS MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif "EWS" in page:
    st.markdown("### Early Warning System Monitor")
    t1, t2 = st.tabs(["Single Account Check", "Portfolio Scan"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            an   = st.text_input("Account Name", "ACE MOBILITY LIMITED")
            dsp2 = st.number_input("Days Since Last Payment", 0, 2800, 38)
            be2  = st.slider("Balance Erosion", 0.1, 2.0, 0.92, 0.01)
        with c2:
            la2  = st.number_input("Loan Age (months)", 0, 120, 22)
            ltv2 = st.slider("LTV", 0.0, 3.0, 1.20, 0.05)
            run  = st.button("🚦 Check EWS", use_container_width=True, type="primary")

        if run:
            ew = ews_check(dsp2, be2, la2, ltv2)
            em = {"RED":"🔴","AMBER":"🟠","YELLOW":"🟡","GREEN":"🟢"}.get(ew["ews_flag"],"🟢")
            msg = f"{em} {ew['ews_flag']} — Score {ew['ews_score']:.0f}/100"
            if   ew["ews_flag"] == "RED":             st.error(msg)
            elif ew["ews_flag"] in ("AMBER","YELLOW"): st.warning(msg)
            else:                                      st.success(msg)
            st.write(f"**Action:** {ew['ews_action']}")
            for t in ew["triggers"]:
                st.warning(f"⚑ {t}")
            if not ew["triggers"]:
                st.success("✓ No triggers fired. Account appears healthy.")

            fig_e = go.Figure(go.Indicator(
                mode="gauge+number", value=ew["ews_score"],
                title={"text": an, "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 100]}, "bar": {"color": NAVY},
                    "steps": [
                        {"range": [0,  25], "color": "#DCFCE7"},
                        {"range": [25, 45], "color": "#FEF9C3"},
                        {"range": [45, 70], "color": "#FFEDD5"},
                        {"range": [70,100], "color": "#FEE2E2"},
                    ],
                }
            ))
            fig_e.update_layout(height=230, margin=dict(t=40,b=0,l=20,r=20),
                                  paper_bgcolor="white")
            st.plotly_chart(fig_e, use_container_width=True)

    with t2:
        st.markdown("### Portfolio EWS Scan")
        st.caption("Upload your monthly loan book to scan every account and get a RED / AMBER / GREEN action list.")

        ews_file = st.file_uploader("Upload loan book (.xlsx)", type=["xlsx"], key="ews_upload")

        if ews_file:
            with st.spinner("Scanning all accounts — please wait..."):
                try:
                    ews_bytes = ews_file.read()
                    xl = pd.ExcelFile(io.BytesIO(ews_bytes))
                    sheet = xl.sheet_names[1] if len(xl.sheet_names) > 1 else xl.sheet_names[0]
                    ews_df = None
                    for hdr in [4, 3, 2, 0]:
                        try:
                            tmp = pd.read_excel(io.BytesIO(ews_bytes), sheet_name=sheet, header=hdr)
                            tmp = tmp.iloc[:, :70].dropna(how="all")
                            cols_lower = [str(c).lower() for c in tmp.columns]
                            if any("balance" in c or "amount" in c for c in cols_lower):
                                ews_df = tmp; break
                        except Exception:
                            continue
                    if ews_df is None:
                        ews_df = pd.read_excel(io.BytesIO(ews_bytes), sheet_name=sheet, header=4)

                    ews_df.columns = [str(c).strip() for c in ews_df.columns]

                    # Get key columns safely
                    def safe_col(df, names, default=0):
                        for n in names:
                            if n in df.columns:
                                return pd.to_numeric(df[n], errors="coerce").fillna(default)
                        return pd.Series([default]*len(df))

                    # Reset index to ensure all series align
                    ews_df = ews_df.reset_index(drop=True)
                    n = len(ews_df)

                    os_bal   = safe_col(ews_df, ["OS Balance","Outstanding Balance","Balance"], 0).reset_index(drop=True)
                    loan_amt = safe_col(ews_df, ["Loan Amount","Loan amount"], 100000).reset_index(drop=True)
                    arr_days = safe_col(ews_df, ["Arrears Days","Days in Arrears"], 0).reset_index(drop=True)
                    coll_amt = safe_col(ews_df, ["Collateral Amount","Security Value"], 0).reset_index(drop=True)

                    REPORT = pd.Timestamp("2026-05-31")
                    if "Disbursed On" in ews_df.columns:
                        disb = pd.to_datetime(ews_df["Disbursed On"], errors="coerce")
                        loan_age = ((REPORT - disb).dt.days / 30.44).fillna(12.0).clip(0,120).reset_index(drop=True)
                    else:
                        loan_age = pd.Series([12.0]*n)

                    if "Last Repayment Date" in ews_df.columns:
                        last_pay = pd.to_datetime(ews_df["Last Repayment Date"], errors="coerce")
                        days_since = ((REPORT - last_pay).dt.days).fillna(90).clip(0,2800).reset_index(drop=True)
                    else:
                        days_since = arr_days.clip(0,2800)

                    # Compute derived signals safely element-wise
                    bal_erosion = pd.Series([
                        float(os_bal.iloc[i] / loan_amt.iloc[i]) if loan_amt.iloc[i] > 0 else 1.0
                        for i in range(n)
                    ]).clip(0, 2)

                    ltv_vals = pd.Series([
                        float(os_bal.iloc[i] / coll_amt.iloc[i]) if coll_amt.iloc[i] > 0 else 2.0
                        for i in range(n)
                    ]).clip(0, 5)

                    # Run EWS on every account
                    results = []
                    for i in range(n):
                        e = ews_check(
                            float(days_since.iloc[i]),
                            float(bal_erosion.iloc[i]),
                            float(loan_age.iloc[i]),
                            float(ltv_vals.iloc[i])
                        )
                        results.append(e)

                    ews_df["EWS Score"] = [r["ews_score"] for r in results]
                    ews_df["EWS Flag"]  = [r["ews_flag"]  for r in results]
                    ews_df["Action"]    = [r["ews_action"] for r in results]

                except Exception as ex:
                    st.error(f"Error reading file: {ex}")
                    st.stop()

            # Summary banners
            total = len(ews_df)
            for flag, color, emoji in [
                ("RED",    "#FEE2E2", "🔴"),
                ("AMBER",  "#FFEDD5", "🟠"),
                ("YELLOW", "#FEF9C3", "🟡"),
                ("GREEN",  "#DCFCE7", "🟢"),
            ]:
                sub = ews_df[ews_df["EWS Flag"] == flag]
                if len(sub) > 0:
                    os_v = os_bal[sub.index].sum()
                    st.markdown(
                        f'<div style="background:{color};padding:10px 16px;border-radius:8px;'
                        f'margin-bottom:6px;font-size:13px;font-weight:600">'
                        f'{emoji} {flag} — {len(sub)} accounts '
                        f'({len(sub)/total*100:.0f}%)  |  '
                        f'OS Balance: KES {os_v/1e6:.1f}M</div>',
                        unsafe_allow_html=True
                    )

            st.divider()

            # Action required accounts
            urgent = ews_df[ews_df["EWS Flag"].isin(["RED","AMBER"])].copy()
            urgent_sorted = urgent.sort_values("EWS Score", ascending=False)

            if len(urgent) > 0:
                st.subheader(f"⚑ {len(urgent)} Accounts Requiring Immediate Action")

                # Show key columns only
                display_cols = []
                for candidate in ["Account Name","Client Name","Account ID",
                                   "Product Name","OS Balance","Arrears Days",
                                   "Classification","EWS Score","EWS Flag","Action"]:
                    if candidate in urgent_sorted.columns:
                        display_cols.append(candidate)

                st.dataframe(
                    urgent_sorted[display_cols].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )

                # Download button
                buf = io.BytesIO()
                urgent_sorted[display_cols].to_excel(buf, index=False)
                st.download_button(
                    "⬇️ Download Action List (.xlsx)",
                    buf.getvalue(),
                    "EWS_Action_List.xlsx",
                    use_container_width=True
                )
            else:
                st.success("✅ No accounts currently in RED or AMBER. Portfolio looks healthy.")

            # Full results download
            st.divider()
            buf2 = io.BytesIO()
            ews_df[["EWS Score","EWS Flag","Action"]].join(
                ews_df.drop(columns=["EWS Score","EWS Flag","Action"])
            ).to_excel(buf2, index=False)
            st.download_button(
                "⬇️ Download Full EWS Report (.xlsx)",
                buf2.getvalue(),
                "Full_EWS_Report.xlsx",
                use_container_width=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown("### Batch Application Processing")

    tpl = pd.DataFrame({
        "applicant_name":    ["SAMPLE CLIENT"],
        "loan_amount":       [500000],
        "repayment_term":    [12],
        "sector":            ["Wholesale and Retail"],
        "product":           ["Chemsha Biashara BC Logbook Loan"],
        "collateral_amount": [750000],
        "days_since_payment":[0],
        "balance_erosion":   [1.0],
        "existing_loan_age": [0],
    })
    st.info("Upload a CSV matching this template:")
    st.dataframe(tpl, use_container_width=True, hide_index=True)
    buf_t = io.BytesIO()
    tpl.to_csv(buf_t, index=False)
    st.download_button("⬇️ Download Template CSV", buf_t.getvalue(), "template.csv")

    st.divider()
    bf = st.file_uploader("Upload applications CSV", type=["csv"])

    if bf:
        apps = pd.read_csv(bf)
        st.write(f"**{len(apps)} applications loaded.**")

        if st.button("▶️ RUN BATCH APPRAISAL", type="primary", use_container_width=True):
            prog = st.progress(0)
            res  = []
            for i, row in apps.iterrows():
                r = full_appraisal(
                    str(row.get("applicant_name", "Unknown")),
                    float(row.get("loan_amount", 100000)),
                    int(row.get("repayment_term", 12)),
                    str(row.get("sector", "Wholesale and Retail")),
                    str(row.get("product", "Boya Discounted Working Capital")),
                    float(row.get("collateral_amount", 0)),
                    float(row.get("existing_loan_age", 0)),
                    float(row.get("days_since_payment", 0)),
                    float(row.get("balance_erosion", 1.0)),
                )
                res.append(r)
                prog.progress((i + 1) / len(apps))

            rdf = pd.DataFrame(res)
            prog.empty()

            app_n = rdf["decision"].str.contains("APPROVE").sum()
            dec_n = rdf["decision"].str.contains("DECLINE").sum()
            ref_n = len(rdf) - app_n - dec_n
            vol   = rdf.loc[rdf["decision"].str.contains("APPROVE"), "loan_amount"].sum()

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Approved",      str(app_n), f"KES {vol/1e6:.1f}M")
            m2.metric("Referred",      str(ref_n))
            m3.metric("Declined",      str(dec_n))
            m4.metric("Approval Rate", f"{app_n/len(rdf):.0%}")

            disp = ["applicant_name","loan_amount","score","grade",
                    "pd_value","recommended_rate","raroc","ews_flag","decision"]
            st.dataframe(rdf[disp], use_container_width=True, hide_index=True)

            buf_o = io.BytesIO()
            rdf[disp].to_excel(buf_o, index=False)
            st.download_button("⬇️ Download Results", buf_o.getvalue(),
                               "batch_results.xlsx", use_container_width=True)

            fig_d = px.histogram(rdf, x="score", nbins=20, title="Score Distribution",
                                  color_discrete_sequence=[NAVY],
                                  labels={"score": "Credit Score"})
            fig_d.add_vline(x=650, line_dash="dash", line_color="#16A34A",
                             annotation_text="Auto-approve (650)")
            fig_d.add_vline(x=580, line_dash="dash", line_color="#CA8A04",
                             annotation_text="Refer (580)")
            fig_d.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_d, use_container_width=True)
