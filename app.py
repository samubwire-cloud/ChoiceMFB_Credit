"""
=============================================================================
  CHOICE MICROFINANCE BANK LIMITED
  Credit Intelligence Platform v2.0
  Full branded Windows application
  Author: Samuel (Head of Credit)
=============================================================================
"""

import streamlit as st
import pandas    as pd
import numpy     as np
import joblib, base64, warnings, io
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express        as px
import plotly.graph_objects  as go
warnings.filterwarnings('ignore')

# ── LOGO (loaded from file or fallback) ─────────────────────────────────────
import os as _os, base64 as _b64
def _load_logo():
    for p in ["choice_logo.png", "/mount/src/choice-mfb-credit/choice_logo.png"]:
        if _os.path.exists(p):
            with open(p,"rb") as f:
                return _b64.b64encode(f.read()).decode()
    return ""
LOGO_B64 = _load_logo()

# ── BRAND COLOURS ───────────────────────────────────────────────────────────
NAVY   = "#110837"
RED    = "#DA2A2F"
LIGHT  = "#F5F6FA"
WHITE  = "#FFFFFF"
GREY   = "#6B7280"
BORDER = "#E5E7EB"

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Choice MFB — Credit Intelligence",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Reset & Base ── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {LIGHT} !important;
}}
[data-testid="stSidebar"] {{
    background: {NAVY} !important;
    border-right: none !important;
}}
[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    color: #CBD5E0 !important;
    font-size: 13px !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    color: {WHITE} !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.15) !important;
}}

/* ── Top header bar ── */
.top-bar {{
    background: {NAVY};
    padding: 14px 24px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
}}
.top-bar-right {{
    text-align: right;
}}
.top-bar-title {{
    font-size: 18px;
    font-weight: 700;
    color: {WHITE};
    margin: 0;
}}
.top-bar-sub {{
    font-size: 11px;
    color: rgba(255,255,255,0.6);
    margin: 0;
}}

/* ── KPI Cards ── */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    margin-bottom: 18px;
}}
.kpi-card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px;
    border-top: 3px solid {NAVY};
}}
.kpi-card.red-top  {{ border-top-color: {RED}; }}
.kpi-card.green-top{{ border-top-color: #16A34A; }}
.kpi-label {{
    font-size: 11px;
    color: {GREY};
    font-weight: 500;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.kpi-value {{
    font-size: 22px;
    font-weight: 700;
    color: {NAVY};
    line-height: 1;
}}
.kpi-sub {{
    font-size: 10px;
    color: {GREY};
    margin-top: 3px;
}}

/* ── Decision banners ── */
.decision-box {{
    border-radius: 10px;
    padding: 14px 20px;
    margin: 12px 0;
    font-size: 16px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.dec-approve  {{ background: #DCFCE7; color: #14532D; border: 1.5px solid #16A34A; }}
.dec-refer    {{ background: #FEF9C3; color: #713F12; border: 1.5px solid #CA8A04; }}
.dec-decline  {{ background: #FEE2E2; color: #7F1D1D; border: 1.5px solid {RED}; }}

/* ── EWS badges ── */
.ews-green  {{ background:#DCFCE7; color:#14532D; padding:6px 14px; border-radius:6px; font-weight:600; display:inline-block; }}
.ews-yellow {{ background:#FEF9C3; color:#713F12; padding:6px 14px; border-radius:6px; font-weight:600; display:inline-block; }}
.ews-amber  {{ background:#FFEDD5; color:#7C2D12; padding:6px 14px; border-radius:6px; font-weight:600; display:inline-block; }}
.ews-red    {{ background:#FEE2E2; color:#7F1D1D; padding:6px 14px; border-radius:6px; font-weight:600; display:inline-block; }}

/* ── Cards / panels ── */
.panel {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 14px;
}}
.panel-title {{
    font-size: 13px;
    font-weight: 600;
    color: {NAVY};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid {RED};
    display: inline-block;
}}

/* ── Section divider ── */
.section-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {GREY};
    margin-bottom: 8px;
}}

/* ── Table styling ── */
.styled-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}}
.styled-table th {{
    background: {NAVY};
    color: {WHITE};
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
}}
.styled-table td {{
    padding: 7px 12px;
    border-bottom: 1px solid {BORDER};
}}
.styled-table tr:nth-child(even) td {{ background: {LIGHT}; }}
.styled-table tr:last-child td {{ font-weight: 600; background: #EEF2FF; }}

/* ── Tip boxes ── */
.tip-box {{
    background: #EFF6FF;
    border-left: 3px solid #3B82F6;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 12px;
    color: #1E40AF;
    margin: 8px 0;
}}
.warn-box {{
    background: #FFFBEB;
    border-left: 3px solid #F59E0B;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 12px;
    color: #92400E;
    margin: 8px 0;
}}

/* ── Streamlit overrides ── */
.stButton > button {{
    background: {NAVY} !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: background 0.2s !important;
}}
.stButton > button:hover {{
    background: #1e0f5e !important;
}}
div[data-testid="stFormSubmitButton"] > button {{
    background: {RED} !important;
    color: {WHITE} !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    border: none !important;
    width: 100% !important;
}}
div[data-testid="stFormSubmitButton"] > button:hover {{
    background: #b91c1c !important;
}}
.stMetric label {{ font-size: 11px !important; color: {GREY} !important; }}
.stMetric [data-testid="stMetricValue"] {{ font-size: 20px !important; color: {NAVY} !important; font-weight: 700 !important; }}
input, select, textarea {{ border-radius: 6px !important; }}
[data-testid="stFileUploader"] {{ border: 2px dashed {BORDER}; border-radius: 10px; padding: 16px; }}
</style>
""", unsafe_allow_html=True)

# ── BANK CONFIGURATION ────────────────────────────────────────────────────────
BANK_CONFIG = dict(
    cof=0.12, opex=0.04, target_profit=0.03, capital_ratio=0.08,
    hurdle_raroc=0.15, lgd_default=0.60, lgd_logbook=0.35,
    lgd_asset_fin=0.40, lgd_unsecured=0.75, lgd_payroll=0.40,
    score_auto_approve=650, score_refer=580,
    max_loan_amount=5_000_000, min_loan_amount=10_000,
    max_ltv=0.80, current_portfolio_os=746_600_000,
    pricing_matrix=dict(
        AAA=dict(score_min=750,std_rate=0.23,floor=0.21,ceiling=0.25),
        AA =dict(score_min=700,std_rate=0.25,floor=0.23,ceiling=0.28),
        A  =dict(score_min=650,std_rate=0.28,floor=0.25,ceiling=0.32),
        BBB=dict(score_min=620,std_rate=0.33,floor=0.28,ceiling=0.38),
        BB =dict(score_min=580,std_rate=0.40,floor=0.33,ceiling=0.48),
        B  =dict(score_min=500,std_rate=0.55,floor=0.45,ceiling=0.65),
        C  =dict(score_min=  0,std_rate=0.72,floor=0.65,ceiling=0.72),
    )
)
PRODUCT_LGD = {
    "Chemsha Biashara BC Logbook Loan"  : 0.35,
    "Chemsha Biashara BC Asset Finance" : 0.40,
    "Staff Development Loan"            : 0.40,
    "Salary Advance"                    : 0.40,
    "Choice Unsecured Loan"             : 0.75,
    "Boya Discounted Working Capital"   : 0.60,
}
BASE_SCORE=600; PDO=20; FACTOR=PDO/np.log(2); OFFSET=BASE_SCORE

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        sc  = joblib.load("scorecard_model.pkl")
        ews = joblib.load("ews_model.pkl")
        return sc, ews, True
    except:
        np.random.seed(42); n=500
        X = pd.DataFrame({f'w_{c}': np.random.normal(0,0.5,n)
                           for c in ['ba','bt','br','bg','bs']})
        y = ((X['w_bg']+X['w_ba']+np.random.normal(0,0.3,n))>0.3).astype(int)
        lr = LogisticRegression(random_state=42,max_iter=500)
        lr.fit(X,y)
        sc = dict(model=lr,
                  woe_maps={c:{} for c in ['ba','bt','br','bg','bs']},
                  bin_cols=['ba','bt','br','bg','bs'],
                  woe_feats=['w_ba','w_bt','w_br','w_bg','w_bs'])
        Xe = pd.DataFrame({
            'sig_days_since_payment': np.abs(np.random.normal(20,40,n)),
            'sig_balance_erosion'   : np.random.uniform(0.3,1.2,n),
            'sig_loan_age'          : np.abs(np.random.normal(10,12,n)),
            'sig_ltv'               : np.random.uniform(0.3,2.5,n),
        })
        rf = RandomForestClassifier(n_estimators=50,random_state=42,class_weight='balanced')
        rf.fit(Xe, y)
        ews = dict(model=rf, features=list(Xe.columns))
        return sc, ews, False

sc_bundle, ews_bundle, prod_mode = load_models()
sc_model  = sc_bundle["model"]
woe_maps  = sc_bundle["woe_maps"]
BIN_COLS  = sc_bundle["bin_cols"]
ews_model = ews_bundle["model"]

# ── ENGINE FUNCTIONS ──────────────────────────────────────────────────────────
def bv(v, t):
    if t=='amount':
        if pd.isna(v) or v<=100000: return 'A'
        elif v<=300000: return 'B'
        elif v<=500000: return 'C'
        elif v<=1000000: return 'D'
        elif v<=2000000: return 'E'
        else: return 'F'
    elif t=='term':
        if pd.isna(v) or v<=3: return 'A'
        elif v<=6: return 'B'
        elif v<=12: return 'C'
        elif v<=18: return 'D'
        elif v<=24: return 'E'
        else: return 'F'
    elif t=='rate':
        if pd.isna(v) or v==0: return 'A'
        elif v<=24: return 'B'
        elif v<=36: return 'C'
        elif v<=48: return 'D'
        else: return 'E'
    elif t=='age':
        if pd.isna(v): return 'Unknown'
        elif v<=6: return 'A'
        elif v<=12: return 'B'
        elif v<=24: return 'C'
        elif v<=48: return 'D'
        else: return 'E'
    elif t=='sector':
        if pd.isna(v): return 'C'
        s=str(v)
        if any(h in s for h in ['Communication Network','Transport infrastructure']): return 'B'
        if any(l in s for l in ['Education','Professional Bodies','Home Development',
                                  'Home Improvements','Real Estate']): return 'A'
        return 'C'

def score_app(la, rt, ir=0, age=0, sec="Wholesale and Retail"):
    bins={'ba':bv(la,'amount'),'bt':bv(rt,'term'),'br':bv(ir or 0,'rate'),
          'bg':bv(age,'age'),'bs':bv(sec,'sector')}
    woes=[woe_maps[c].get(str(bins[c]),0.0) for c in BIN_COLS]
    X=np.array(woes).reshape(1,-1)
    pd_v=float(sc_model.predict_proba(X)[0,1])
    p=np.clip(pd_v,0.0001,0.9999)
    sc=int(np.clip(OFFSET+FACTOR*np.log((1-p)/p),300,850))
    gr="C"
    for g,pm in BANK_CONFIG["pricing_matrix"].items():
        if sc>=pm["score_min"]: gr=g; break
    return dict(pd_value=round(pd_v,4), score=sc, grade=gr)

def price_l(pd_v, la, col=0, prod="Standard"):
    COF=BANK_CONFIG["cof"]; OPX=BANK_CONFIG["opex"]
    PRF=BANK_CONFIG["target_profit"]; CAP=BANK_CONFIG["capital_ratio"]
    lgd=PRODUCT_LGD.get(prod, BANK_CONFIG["lgd_default"])
    if prod not in PRODUCT_LGD:
        ltv=col/la if col>0 else 0
        lgd=0.20 if ltv>=1.5 else 0.35 if ltv>=1.0 else 0.50 if ltv>=0.5 else 0.65 if ltv>0 else 0.75
    el_r=pd_v*lgd; el_k=el_r*la
    rbp=float(np.clip(COF+OPX+el_r+PRF,0.19,0.72))
    sc2=int(np.clip(OFFSET+FACTOR*np.log((1-np.clip(pd_v,0.0001,0.9999))/np.clip(pd_v,0.0001,0.9999)),300,850))
    gr_r=0.72
    for g,pm in BANK_CONFIG["pricing_matrix"].items():
        if sc2>=pm["score_min"]: gr_r=pm["std_rate"]; break
    rec=round(max(rbp,gr_r),4)
    ni=la*(rec-COF-OPX)-el_k; cap=la*CAP
    raroc=ni/cap if cap>0 else 0
    return dict(lgd=lgd,el_rate=round(el_r,4),el_kes=round(el_k),
                rbp_floor=round(rbp,4),recommended_rate=rec,raroc=round(raroc,4))

def ews_check(dsp=5, be=0.5, lam=0, ltv=0.5):
    feats=np.array([[float(np.clip(dsp,0,2800)),float(np.clip(be,0,2)),
                     float(np.clip(lam,0,120)),float(np.clip(ltv,0,5))]])
    prob=float(ews_model.predict_proba(feats)[0,1])
    sc=round(prob*100,1)
    if sc>=70:   flag="🔴 RED";    act="Immediate escalation — same day"
    elif sc>=45: flag="🟠 AMBER";  act="Proactive contact within 48 hours"
    elif sc>=25: flag="🟡 YELLOW"; act="Schedule check-in within 5 days"
    else:        flag="🟢 GREEN";  act="No action — monthly monitoring"
    trg=[]
    if dsp>=45: trg.append("No payment for 45+ days")
    if dsp>=30: trg.append("No payment for 30+ days")
    if be>=0.95 and lam>6: trg.append("Low repayment velocity")
    if lam>=18: trg.append(f"Loan age {lam:.0f} months — maturity risk")
    if ltv>1.0: trg.append(f"LTV breach ({ltv:.0%})")
    return dict(ews_score=sc, ews_flag=flag, ews_action=act, triggers=trg)

def full_appraisal(name, la, rt, sec, prod, col=0, ir=0, age=0, dsp=0, be=1.0):
    s=score_app(la,rt,ir,age,sec)
    p=price_l(s["pd_value"],la,col,prod)
    ltv_e=col/la if col>0 else 2.0
    e=ews_check(dsp,be,age,ltv_e)
    if   s["score"]>=BANK_CONFIG["score_auto_approve"]: dec="✅ AUTO-APPROVE"; dc="dec-approve"
    elif s["score"]>=BANK_CONFIG["score_refer"]:         dec="✅ APPROVE";      dc="dec-approve"
    elif s["score"]>=500:                                dec="⟳ REFER TO COMMITTEE"; dc="dec-refer"
    else:                                                dec="❌ DECLINE";      dc="dec-decline"
    if e["ews_flag"].startswith("🔴") and s["score"]>=580:
        dec="⟳ REFER — EWS RED FLAG"; dc="dec-refer"
    rm=p["recommended_rate"]/12
    mo=la*rm*(1+rm)**rt/((1+rm)**rt-1) if rm>0 and rt>0 else la/rt if rt>0 else 0
    return {**s,**p,**e,"decision":dec,"dec_class":dc,
            "monthly_payment":round(mo),"applicant_name":name,
            "loan_amount":la,"repayment_term":rt,"collateral_amount":col}

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo in sidebar
    st.markdown(
        f'<div style="background:white;border-radius:8px;padding:10px 12px;margin-bottom:16px;text-align:center">'
        f'<img src="data:image/png;base64,{LOGO_B64}" style="width:100%;max-width:200px"/>'
        f'</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">Navigation</div>', unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Dashboard",
        "📋  Loan Appraisal",
        "📊  Portfolio Analytics",
        "🚦  EWS Monitor",
        "📁  Batch Processing",
        "⚙️  Settings",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    mode_txt = "✅ Production" if prod_mode else "⚠️ Demo mode"
    st.markdown(f'<div style="font-size:11px;color:rgba(255,255,255,0.5)">{mode_txt}<br>Riverside Branch · May 2026</div>',
                unsafe_allow_html=True)
    if not prod_mode:
        st.markdown('<div style="font-size:10px;color:#FBBF24;margin-top:6px">Place .pkl model files<br>in app folder for full accuracy</div>',
                    unsafe_allow_html=True)


# ── HEADER BAR (all pages) ──────────────────────────────────────────────────
st.markdown(f"""
<div class="top-bar">
  <img src="data:image/png;base64,{LOGO_B64}" style="height:36px"/>
  <div class="top-bar-right">
    <p class="top-bar-title">Credit Intelligence Platform</p>
    <p class="top-bar-sub">Samuel Njoroge · Head of Credit · Riverside Branch</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown('<div class="panel-title">Portfolio Overview — May 2026</div>', unsafe_allow_html=True)

    # KPI row
    st.markdown("""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-label">Total Accounts</div><div class="kpi-value">725</div><div class="kpi-sub">Active loans</div></div>
      <div class="kpi-card red-top"><div class="kpi-label">NPL Ratio</div><div class="kpi-value">9.0%</div><div class="kpi-sub">Benchmark: &lt;5%</div></div>
      <div class="kpi-card"><div class="kpi-label">Outstanding</div><div class="kpi-value">KES 746M</div><div class="kpi-sub">Total book</div></div>
      <div class="kpi-card green-top"><div class="kpi-label">Prov. Coverage</div><div class="kpi-value">91%</div><div class="kpi-sub">KES 61.5M</div></div>
      <div class="kpi-card red-top"><div class="kpi-label">PAR 30</div><div class="kpi-value">9.0%</div><div class="kpi-sub">KES 67.5M at risk</div></div>
      <div class="kpi-card"><div class="kpi-label">Active Products</div><div class="kpi-value">23</div><div class="kpi-sub">Loan products</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="panel-title">PAR Aging Distribution</span>', unsafe_allow_html=True)
        par_data = pd.DataFrame({
            "Bucket": ["Current","PAR 1–30d","PAR 31–60d","PAR 61–90d","PAR 91–180d","PAR 181–365d","PAR >365d"],
            "OS_M"  : [224.5, 81.2, 20.5, 8.1, 18.5, 11.0, 9.4],
            "Pct"   : [30.1,  10.9,  2.8, 1.1,  2.5,  1.5,  1.3],
        })
        colors = ["#16A34A","#CA8A04","#EA580C","#DC2626","#991B1B","#7F1D1D","#450A0A"]
        fig = go.Figure(go.Bar(
            x=par_data["Bucket"], y=par_data["OS_M"],
            marker_color=colors,
            text=[f"{p:.1f}%" for p in par_data["Pct"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=280, margin=dict(t=10,b=0,l=0,r=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="KES Millions",
            font=dict(family="Inter, sans-serif", size=11),
            yaxis=dict(gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="panel-title">Loan Classification</span>', unsafe_allow_html=True)
        clf_data = pd.DataFrame({
            "Class": ["Normal","Watch","Substandard","Doubtful","Loss"],
            "Value": [597.8, 81.2, 20.5, 8.1, 38.9],
        })
        fig2 = go.Figure(go.Pie(
            labels=clf_data["Class"],
            values=clf_data["Value"],
            hole=0.45,
            marker_colors=["#16A34A","#CA8A04","#EA580C","#DC2626","#7F1D1D"],
            textinfo="percent+label",
            textfont_size=11,
        ))
        fig2.update_layout(
            height=280, margin=dict(t=10,b=0,l=0,r=0),
            paper_bgcolor="white",
            showlegend=False,
            font=dict(family="Inter, sans-serif"),
            annotations=[dict(text="KES 746M", x=0.5, y=0.5,
                              font_size=12, font_color=NAVY, showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # EWS + Pricing summary row
    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown(f"""<div class="panel">
        <span class="panel-title">EWS Status</span><br><br>
        <div class="ews-red" style="margin-bottom:8px">🔴 RED &nbsp;&nbsp; 168 accounts · KES 60.6M</div><br>
        <div class="ews-amber" style="margin-bottom:8px">🟠 AMBER &nbsp; 13 accounts · KES 1.1M</div><br>
        <div class="ews-yellow" style="margin-bottom:8px">🟡 YELLOW &nbsp;23 accounts</div><br>
        <div class="ews-green">🟢 GREEN &nbsp;&nbsp;521 accounts</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="panel">
        <span class="panel-title">Pricing Health</span><br><br>
        <div style="font-size:13px;margin-bottom:10px">
        <b style="color:{NAVY}">174</b> <span style="color:{GREY}">loans underpriced (RBP &gt; Actual)</span><br><br>
        <b style="color:{RED}">218</b> <span style="color:{GREY}">loans negative RAROC</span><br><br>
        <b style="color:#16A34A">497</b> <span style="color:{GREY}">loans value-creating</span><br><br>
        <b style="color:{NAVY}">KES 42.2M</b> <span style="color:{GREY}">pa income uplift under full RBP</span>
        </div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="panel">
        <span class="panel-title">Top Actions</span><br><br>
        <div style="font-size:12px;line-height:1.9;color:{GREY}">
        ⚑ <b style="color:{RED}">Call 13 AMBER</b> accounts today<br>
        ⚑ <b style="color:{RED}">Write off</b> Platinum Imports (2,719d)<br>
        ⚑ <b style="color:#CA8A04">Reprice</b> Staff Dev Loans (8.4% rate)<br>
        ⚑ <b style="color:#CA8A04">Review</b> 38 underwater loans (LTV&gt;100%)<br>
        ✓ <b style="color:#16A34A">May-26</b> collections exceeded disbursals
        </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LOAN APPRAISAL
# ══════════════════════════════════════════════════════════════════════════════
elif "Appraisal" in page:
    st.markdown('<div class="panel-title">New Loan Appraisal</div>', unsafe_allow_html=True)

    with st.form("appraisal_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-label">Client Details</div>', unsafe_allow_html=True)
            name    = st.text_input("Client Name", placeholder="e.g. WANJIKU TRADERS LTD")
            product = st.selectbox("Product", list(PRODUCT_LGD.keys()) + ["Other"])
            sector  = st.selectbox("Business Sector", [
                "Wholesale and Retail","Transport infrastructure development services",
                "Home Development","Professional Bodies","Education",
                "Communication Network Systems","Real Estate Development","Other",
            ])
        with c2:
            st.markdown('<div class="section-label">Loan Details</div>', unsafe_allow_html=True)
            la  = st.number_input("Loan Amount (KES)", 10_000, 5_000_000, 500_000, 10_000, format="%d")
            rt  = st.selectbox("Repayment Term (months)", [3,6,9,12,18,24,36,48,60], index=3)
            col = st.number_input("Collateral Value (KES)", 0, 20_000_000, 0, 50_000, format="%d")
        with c3:
            st.markdown('<div class="section-label">Relationship (0 for new clients)</div>', unsafe_allow_html=True)
            age = st.number_input("Existing Loan Age (months)", 0, 120, 0)
            dsp = st.number_input("Days Since Last Payment",    0, 2800, 0)
            be  = st.slider("Balance Erosion (1.0 = new)", 0.1, 2.0, 1.0, 0.05)

        submitted = st.form_submit_button("🔍 RUN CREDIT APPRAISAL", use_container_width=True)

    if submitted:
        if not name:
            st.error("Please enter a client name.")
        else:
            with st.spinner("Running appraisal..."):
                r = full_appraisal(name, la, rt, sector, product, col, 0, age, dsp, be)

            # Decision banner
            st.markdown(f'<div class="decision-box {r["dec_class"]}">{r["decision"]}</div>',
                        unsafe_allow_html=True)

            # KPIs
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            k1.metric("Credit Score",   f"{r['score']} / 850")
            k2.metric("Grade",          r["grade"])
            k3.metric("PD (1-year)",    f"{r['pd_value']:.1%}")
            k4.metric("Rec. Rate",      f"{r['recommended_rate']:.1%}")
            k5.metric("RAROC",          f"{r['raroc']:.1%}")
            k6.metric("Monthly PMT",    f"KES {r['monthly_payment']:,.0f}")

            ca, cb = st.columns(2)
            with ca:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<span class="panel-title">Risk-Based Pricing</span>', unsafe_allow_html=True)
                ltv = col/la if col>0 else 0
                st.markdown(f"""
                <table class="styled-table">
                <tr><th>Component</th><th>Rate</th></tr>
                <tr><td>Cost of Funds (CoF)</td><td>{BANK_CONFIG['cof']:.1%}</td></tr>
                <tr><td>Operating Expenses</td><td>{BANK_CONFIG['opex']:.1%}</td></tr>
                <tr><td>Expected Loss (PD {r['pd_value']:.1%} × LGD {r['lgd']:.0%})</td><td>{r['el_rate']:.1%}</td></tr>
                <tr><td>Target Profit</td><td>{BANK_CONFIG['target_profit']:.1%}</td></tr>
                <tr><td>RBP Floor Rate</td><td>{r['rbp_floor']:.1%}</td></tr>
                <tr><td><b>Recommended Rate</b></td><td><b>{r['recommended_rate']:.1%}</b></td></tr>
                </table>
                <div style="font-size:11px;color:{GREY};margin-top:8px">
                LTV: {ltv:.0%} &nbsp;|&nbsp; Annual EL: KES {r['el_kes']:,.0f} &nbsp;|&nbsp; RAROC: {r['raroc']:.1%}
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with cb:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<span class="panel-title">Early Warning Assessment</span>', unsafe_allow_html=True)
                ews_css = ("ews-green"  if "GREEN"  in r["ews_flag"] else
                           "ews-yellow" if "YELLOW" in r["ews_flag"] else
                           "ews-amber"  if "AMBER"  in r["ews_flag"] else "ews-red")
                st.markdown(f'<div class="{ews_css}" style="font-size:16px;margin-bottom:12px">{r["ews_flag"]} — Score {r["ews_score"]:.0f}/100</div>',
                            unsafe_allow_html=True)
                st.write(f"**Action:** {r['ews_action']}")
                if r["triggers"]:
                    for t in r["triggers"]:
                        st.markdown(f'<div class="warn-box">⚑ {t}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="tip-box">✓ No warning triggers. Loan appears healthy.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Score gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r["score"],
                title={"text":f"{name} — Grade {r['grade']}", "font":{"size":13,"color":NAVY}},
                gauge={
                    "axis":{"range":[300,850],"tickwidth":1,"tickcolor":GREY},
                    "bar":{"color":NAVY},
                    "bgcolor":"white",
                    "steps":[
                        {"range":[300,500],"color":"#FEE2E2"},
                        {"range":[500,580],"color":"#FFEDD5"},
                        {"range":[580,650],"color":"#FEF9C3"},
                        {"range":[650,750],"color":"#DCFCE7"},
                        {"range":[750,850],"color":"#BBF7D0"},
                    ],
                    "threshold":{"line":{"color":RED,"width":3},"value":580}
                }
            ))
            fig_g.update_layout(height=250, margin=dict(t=40,b=0,l=20,r=20),
                                  paper_bgcolor="white",
                                  font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig_g, use_container_width=True)

            # Memo download
            ltv_d = col/la if col>0 else 0
            memo = f"CHOICE MICROFINANCE BANK LIMITED\nCREDIT APPRAISAL MEMO\n{'='*50}\nClient: {name}\nProduct: {product}\nSector: {sector}\n{'─'*50}\nLoan: KES {la:,.0f} | Term: {rt} months | Collateral: KES {col:,.0f} | LTV: {ltv_d:.0%}\n{'─'*50}\nScore: {r['score']}/850 | Grade: {r['grade']} | PD: {r['pd_value']:.1%}\nEWS: {r['ews_flag']} ({r['ews_score']:.0f}/100)\n{'─'*50}\nRBP Floor: {r['rbp_floor']:.1%} | Recommended Rate: {r['recommended_rate']:.1%}\nRAROC: {r['raroc']:.1%} | Monthly Payment: KES {r['monthly_payment']:,.0f}\n{'─'*50}\nDECISION: {r['decision']}\n{'='*50}"
            st.download_button("⬇️ Download Credit Memo (.txt)",
                               memo.encode(), f"Memo_{name.replace(' ','_')}.txt",
                               use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown('<div class="panel-title">Portfolio Analytics</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload monthly loan book (.xlsx)", type=["xlsx"])

    @st.cache_data
    def load_book(fb):
        df=pd.read_excel(io.BytesIO(fb), sheet_name=1, header=4)
        df=df.iloc[:,:58].dropna(how='all')
        for c in ['Loan Amount','OS Balance','Arrears Days','Arrears Amount',
                  'Interest Rate','Provision','Collateral Amount','Installment Amount','Repayment Term']:
            df[c]=pd.to_numeric(df[c],errors='coerce')
        df['Disbursed On']=pd.to_datetime(df['Disbursed On'],errors='coerce')
        cm={'Normal':'Normal','Watch':'Watch','Substandard':'Substandard',
            'doubtful':'Doubtful','loss':'Loss',0:'Normal'}
        df['Classification']=df['Classification'].map(cm).fillna('Unknown')
        R=pd.Timestamp('2026-05-31')
        df['loan_age']=(R-df['Disbursed On']).dt.days/30.44
        df['bad']=df['Classification'].isin(['Substandard','Doubtful','Loss']).astype(int)
        def pb(d):
            if pd.isna(d): return 'Unknown'
            elif d==0: return '1.Current'
            elif d<=30: return '2.PAR 1-30d'
            elif d<=60: return '3.PAR 31-60d'
            elif d<=90: return '4.PAR 61-90d'
            elif d<=180: return '5.PAR 91-180d'
            elif d<=365: return '6.PAR 181-365d'
            else: return '7.PAR >365d'
        df['par_bucket']=df['Arrears Days'].apply(pb)
        return df

    if up:
        df=load_book(up.read())
        total_os=df['OS Balance'].sum()
        npl_v=df.loc[df['bad']==1,'OS Balance'].sum()
        prov=df['Provision'].sum()

        st.markdown(f"""<div class="kpi-row">
        <div class="kpi-card"><div class="kpi-label">Accounts</div><div class="kpi-value">{len(df):,}</div></div>
        <div class="kpi-card"><div class="kpi-label">OS Balance</div><div class="kpi-value">KES {total_os/1e6:.1f}M</div></div>
        <div class="kpi-card red-top"><div class="kpi-label">NPL Ratio</div><div class="kpi-value">{npl_v/total_os*100:.1f}%</div></div>
        <div class="kpi-card green-top"><div class="kpi-label">Prov. Coverage</div><div class="kpi-value">{prov/npl_v*100:.0f}%</div></div>
        <div class="kpi-card"><div class="kpi-label">Bad Rate</div><div class="kpi-value">{df['bad'].mean():.1%}</div></div>
        <div class="kpi-card"><div class="kpi-label">Products</div><div class="kpi-value">{df['Product Name'].nunique()}</div></div>
        </div>""", unsafe_allow_html=True)

        c1,c2=st.columns(2)
        with c1:
            par_s=df.groupby('par_bucket')['OS Balance'].sum().reset_index()
            par_s['pct']=par_s['OS Balance']/total_os*100
            fig=px.bar(par_s,x='par_bucket',y='OS Balance',
                       text=par_s['pct'].apply(lambda x:f"{x:.1f}%"),
                       color_discrete_sequence=[NAVY],
                       labels={'par_bucket':'','OS Balance':'KES'}, title="PAR Aging")
            fig.update_layout(height=300,plot_bgcolor='white',paper_bgcolor='white',
                               yaxis=dict(gridcolor='#F3F4F6'),
                               font=dict(family='Inter, sans-serif',size=11))
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            clf_s=df.groupby('Classification')['OS Balance'].sum().reset_index()
            cc={'Normal':'#16A34A','Watch':'#CA8A04','Substandard':'#EA580C',
                'Doubtful':'#DC2626','Loss':'#7F1D1D','Unknown':'#9CA3AF'}
            fig2=px.pie(clf_s,values='OS Balance',names='Classification',
                        color='Classification',color_discrete_map=cc,
                        hole=0.4,title="Classification")
            fig2.update_layout(height=300,paper_bgcolor='white',showlegend=True,
                                font=dict(family='Inter, sans-serif'))
            st.plotly_chart(fig2,use_container_width=True)

        prod_s=df.groupby('Product Name').agg(
            count=('bad','count'),npl_rate=('bad','mean'),os=('OS Balance','sum')).reset_index()
        prod_s=prod_s[prod_s['count']>=3].sort_values('os',ascending=False)
        prod_s['npl_pct']=prod_s['npl_rate']*100
        fig3=px.bar(prod_s,x='Product Name',y='npl_pct',
                    color='npl_pct',
                    color_continuous_scale=[(0,'#16A34A'),(0.2,'#CA8A04'),(0.5,'#EA580C'),(1,'#7F1D1D')],
                    text=prod_s['npl_pct'].apply(lambda x:f"{x:.0f}%"),
                    labels={'npl_pct':'NPL Rate (%)','Product Name':''},
                    title="NPL Rate by Product")
        fig3.update_layout(height=320,plot_bgcolor='white',paper_bgcolor='white',
                            xaxis_tickangle=-30,
                            font=dict(family='Inter, sans-serif',size=11))
        fig3.update_traces(textposition='outside')
        st.plotly_chart(fig3,use_container_width=True)
    else:
        st.markdown('<div class="tip-box">📂 Upload your monthly BSM loan book Excel file to see live analytics.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EWS MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif "EWS" in page:
    st.markdown('<div class="panel-title">Early Warning System Monitor</div>', unsafe_allow_html=True)
    t1,t2=st.tabs(["Single Account","Portfolio Scan"])

    with t1:
        c1,c2=st.columns(2)
        with c1:
            an  =st.text_input("Account Name","ACE MOBILITY LIMITED")
            dsp2=st.number_input("Days Since Last Payment",0,2800,38)
            be2 =st.slider("Balance Erosion",0.1,2.0,0.92,0.01)
        with c2:
            la2 =st.number_input("Loan Age (months)",0,120,22)
            ltv2=st.slider("LTV",0.0,3.0,1.20,0.05)
            go_ews=st.button("🚦 Check EWS",use_container_width=True,type="primary")
        if go_ews:
            ew=ews_check(dsp2,be2,la2,ltv2)
            ews_c=("ews-green" if "GREEN" in ew["ews_flag"] else
                   "ews-yellow" if "YELLOW" in ew["ews_flag"] else
                   "ews-amber" if "AMBER" in ew["ews_flag"] else "ews-red")
            st.markdown(f'<div class="{ews_c}" style="font-size:18px;margin:12px 0">{ew["ews_flag"]} — Score {ew["ews_score"]:.0f}/100</div>',
                        unsafe_allow_html=True)
            st.write(f"**Action:** {ew['ews_action']}")
            for t in ew["triggers"]:
                st.markdown(f'<div class="warn-box">⚑ {t}</div>',unsafe_allow_html=True)
            if not ew["triggers"]:
                st.markdown('<div class="tip-box">✓ No triggers. Account appears healthy.</div>',unsafe_allow_html=True)
            fig_e=go.Figure(go.Indicator(
                mode="gauge+number",value=ew["ews_score"],
                title={"text":an,"font":{"size":13,"color":NAVY}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":NAVY},
                       "steps":[{"range":[0,25],"color":"#DCFCE7"},
                                 {"range":[25,45],"color":"#FEF9C3"},
                                 {"range":[45,70],"color":"#FFEDD5"},
                                 {"range":[70,100],"color":"#FEE2E2"}]}
            ))
            fig_e.update_layout(height=230,margin=dict(t=40,b=0,l=20,r=20),paper_bgcolor="white")
            st.plotly_chart(fig_e,use_container_width=True)

    with t2:
        ef=st.file_uploader("Upload loan book for EWS scan",type=["xlsx"],key="ews2")
        if ef:
            with st.spinner("Scanning portfolio..."):
                df2=load_book(ef.read()) if 'load_book' in dir() else pd.DataFrame()
                if len(df2)>0:
                    R=pd.Timestamp('2026-05-31')
                    df2['dsp']=((R-pd.to_datetime(df2.get('Last Repayment Date'),errors='coerce')).dt.days).clip(lower=0).fillna(90)
                    df2['be_r']=np.where(df2['Loan Amount']>0,df2['OS Balance']/df2['Loan Amount'],1.0).clip(0,2)
                    df2['ltv_v']=np.where(df2['Collateral Amount']>0,df2['OS Balance']/df2['Collateral Amount'],2.0).clip(0,5)
                    df2['lam']=df2['loan_age'].fillna(0)
                    ews_r=[ews_check(r['dsp'],r['be_r'],r['lam'],r['ltv_v']) for _,r in df2.iterrows()]
                    df2['ews_score']=[x['ews_score'] for x in ews_r]
                    df2['ews_flag'] =[x['ews_flag']  for x in ews_r]

            for flg,col_style in [("🔴 RED","#FEE2E2"),("🟠 AMBER","#FFEDD5"),
                                   ("🟡 YELLOW","#FEF9C3"),("🟢 GREEN","#DCFCE7")]:
                sub=df2[df2['ews_flag']==flg]
                if len(sub)>0:
                    os_v=sub['OS Balance'].sum()
                    st.markdown(f'<div style="background:{col_style};padding:8px 14px;border-radius:6px;margin-bottom:6px;font-size:13px"><b>{flg}</b> — {len(sub)} accounts | KES {os_v/1e6:.1f}M</div>',unsafe_allow_html=True)
            urgent=df2[df2['ews_flag'].isin(["🔴 RED","🟠 AMBER"])].copy()
            if len(urgent)>0:
                st.markdown(f'<div class="panel-title">{len(urgent)} Accounts Requiring Action</div>',unsafe_allow_html=True)
                show=[c for c in ['Account Name','Product Name','OS Balance','Arrears Days','ews_score','ews_flag','Classification'] if c in urgent.columns]
                st.dataframe(urgent[show].sort_values('ews_score',ascending=False),use_container_width=True,hide_index=True)
                buf=io.BytesIO(); urgent[show].to_excel(buf,index=False)
                st.download_button("⬇️ Download Action List",buf.getvalue(),"EWS_Actions.xlsx",use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown('<div class="panel-title">Batch Application Processing</div>', unsafe_allow_html=True)
    tpl=pd.DataFrame({"applicant_name":["SAMPLE CLIENT"],"loan_amount":[500000],
                       "repayment_term":[12],"sector":["Wholesale and Retail"],
                       "product":["Chemsha Biashara BC Logbook Loan"],
                       "collateral_amount":[750000],"days_since_payment":[0],
                       "balance_erosion":[1.0],"existing_loan_age":[0]})
    st.markdown('<div class="tip-box">Upload a CSV matching this template format:</div>',unsafe_allow_html=True)
    st.dataframe(tpl,use_container_width=True,hide_index=True)
    buf_t=io.BytesIO(); tpl.to_csv(buf_t,index=False)
    st.download_button("⬇️ Download Template CSV",buf_t.getvalue(),"template.csv")
    st.divider()
    bf=st.file_uploader("Upload applications CSV",type=["csv"])
    if bf:
        apps=pd.read_csv(bf)
        st.write(f"**{len(apps)} applications loaded.**")
        if st.button("▶️ RUN BATCH APPRAISAL",type="primary",use_container_width=True):
            prog=st.progress(0); res=[]
            for i,row in apps.iterrows():
                r=full_appraisal(str(row.get("applicant_name","Unknown")),
                                  float(row.get("loan_amount",100000)),
                                  int(row.get("repayment_term",12)),
                                  str(row.get("sector","Wholesale and Retail")),
                                  str(row.get("product","Boya Discounted Working Capital")),
                                  float(row.get("collateral_amount",0)),
                                  0,float(row.get("existing_loan_age",0)),
                                  float(row.get("days_since_payment",0)),
                                  float(row.get("balance_erosion",1.0)))
                res.append(r); prog.progress((i+1)/len(apps))
            rdf=pd.DataFrame(res); prog.empty()
            app_n=(rdf["decision"].str.contains("APPROVE")).sum()
            dec_n=(rdf["decision"].str.contains("DECLINE")).sum()
            ref_n=(rdf["decision"].str.contains("REFER")).sum()
            vol=rdf.loc[rdf["decision"].str.contains("APPROVE"),"loan_amount"].sum()
            st.markdown(f"""<div class="kpi-row" style="grid-template-columns:repeat(4,1fr)">
            <div class="kpi-card green-top"><div class="kpi-label">Approved</div><div class="kpi-value">{app_n}</div><div class="kpi-sub">KES {vol/1e6:.1f}M</div></div>
            <div class="kpi-card"><div class="kpi-label">Referred</div><div class="kpi-value">{ref_n}</div></div>
            <div class="kpi-card red-top"><div class="kpi-label">Declined</div><div class="kpi-value">{dec_n}</div></div>
            <div class="kpi-card"><div class="kpi-label">Approval Rate</div><div class="kpi-value">{app_n/len(rdf):.0%}</div></div>
            </div>""",unsafe_allow_html=True)
            disp=['applicant_name','loan_amount','score','grade','pd_value','recommended_rate','raroc','ews_flag','decision']
            st.dataframe(rdf[disp],use_container_width=True,hide_index=True)
            buf_o=io.BytesIO(); rdf[disp].to_excel(buf_o,index=False)
            st.download_button("⬇️ Download Results",buf_o.getvalue(),"batch_results.xlsx",use_container_width=True)
            fig_d=px.histogram(rdf,x="score",nbins=20,color_discrete_sequence=[NAVY],
                                labels={"score":"Credit Score"},title="Score Distribution")
            fig_d.add_vline(x=650,line_dash="dash",line_color="#16A34A",annotation_text="Auto-approve")
            fig_d.add_vline(x=580,line_dash="dash",line_color="#CA8A04",annotation_text="Refer")
            fig_d.update_layout(height=300,plot_bgcolor='white',paper_bgcolor='white')
            st.plotly_chart(fig_d,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif "Settings" in page:
    st.markdown('<div class="panel-title">System Settings</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="panel-title">Bank Parameters</span>', unsafe_allow_html=True)
        st.number_input("Cost of Funds (% pa)", 8.0, 20.0, BANK_CONFIG['cof']*100, 0.5, format="%.1f")
        st.number_input("OpEx Ratio (% pa)",    2.0, 10.0, BANK_CONFIG['opex']*100, 0.5, format="%.1f")
        st.number_input("Target Profit (% pa)", 1.0,  8.0, BANK_CONFIG['target_profit']*100, 0.5, format="%.1f")
        st.number_input("LGD — Blended (%)",   40.0, 80.0, BANK_CONFIG['lgd_default']*100, 1.0, format="%.0f")
        st.number_input("Auto-approve Score Threshold", 600, 750, BANK_CONFIG['score_auto_approve'])
        st.markdown('<div class="tip-box">Changes here are for reference only. To make permanent changes, update BANK_CONFIG in the app file.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="panel-title">System Information</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:12px;line-height:2.2;color:{GREY}">
        <b style="color:{NAVY}">App Version</b> &nbsp; 2.0.0<br>
        <b style="color:{NAVY}">Model Status</b> &nbsp; {'✅ Production' if prod_mode else '⚠️ Demo mode'}<br>
        <b style="color:{NAVY}">Scorecard AUC</b> &nbsp; 0.769<br>
        <b style="color:{NAVY}">EWS AUC</b> &nbsp; 0.964<br>
        <b style="color:{NAVY}">Built by</b> &nbsp; Samuel (Head of Credit)<br>
        <b style="color:{NAVY}">Branch</b> &nbsp; Riverside · Choice MFB<br>
        <b style="color:{NAVY}">Last trained</b> &nbsp; May 2026<br>
        <b style="color:{NAVY}">Next retraining</b> &nbsp; August 2026
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<span class="panel-title">Model Files</span>', unsafe_allow_html=True)
        import os
        for fn in ['scorecard_model.pkl','ews_model.pkl','choice_mfb_credit_app_v2.py']:
            exists = os.path.exists(fn)
            icon = "✅" if exists else "❌"
            size = f"{os.path.getsize(fn)/1024:.0f} KB" if exists else "Not found"
            st.markdown(f'<div style="font-size:12px;margin-bottom:4px">{icon} <b>{fn}</b> — {size}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
