import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import json
import os
import re

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ---------- Reset & Base ---------- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 20% 10%, rgba(99,102,241,.18) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 80% 80%, rgba(16,185,129,.12) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #13151d !important;
    border-right: 1px solid rgba(255,255,255,.06) !important;
}
[data-testid="stSidebar"] * { color: #c8cad4 !important; }

/* ---------- Cards / containers ---------- */
.card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
}
.card-accent { border-left: 3px solid #6366f1; }

/* ---------- Page title ---------- */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -.02em;
    background: linear-gradient(135deg, #e8eaf0 30%, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.page-sub {
    font-size: .95rem;
    color: #6b7280;
    margin-bottom: 32px;
    font-weight: 300;
}

/* ---------- Section headers ---------- */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a5b4fc;
    letter-spacing: .06em;
    text-transform: uppercase;
    margin: 32px 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,.07);
}

/* ---------- Metric tiles ---------- */
.metric-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
.metric-tile {
    flex: 1;
    min-width: 140px;
    background: rgba(255,255,255,.05);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
}
.metric-tile .val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e8eaf0;
}
.metric-tile .lbl {
    font-size: .78rem;
    color: #6b7280;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: .06em;
}
.metric-tile.green .val { color: #10b981; }
.metric-tile.amber .val { color: #f59e0b; }
.metric-tile.red   .val { color: #ef4444; }

/* ---------- Inputs ---------- */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.25), 0 0 20px rgba(99,102,241,.1) !important;
}
label { color: #9ca3af !important; font-size: .85rem !important; }

/* ---------- Buttons ---------- */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: .95rem !important;
    padding: 12px !important;
    letter-spacing: .03em;
    transition: opacity .2s, transform .15s, box-shadow .2s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: .9;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,.4) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ---------- Alerts ---------- */
[data-testid="stSuccess"] {
    background: rgba(16,185,129,.12) !important;
    border-left: 3px solid #10b981 !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
}
[data-testid="stInfo"] {
    background: rgba(99,102,241,.12) !important;
    border-left: 3px solid #6366f1 !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
}
[data-testid="stWarning"] {
    background: rgba(245,158,11,.10) !important;
    border-left: 3px solid #f59e0b !important;
    border-radius: 10px !important;
    color: #fcd34d !important;
}
[data-testid="stError"] {
    background: rgba(239,68,68,.10) !important;
    border-left: 3px solid #ef4444 !important;
    border-radius: 10px !important;
    color: #fca5a5 !important;
}

/* ---------- Sliders ---------- */
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #a5b4fc !important; }
[data-testid="stSlider"] [class*="Track"] { background: #6366f1 !important; }

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
iframe { border-radius: 12px; }

/* ---------- Suggestion pills ---------- */
.suggestion-pill {
    display: inline-block;
    background: rgba(99,102,241,.15);
    border: 1px solid rgba(99,102,241,.3);
    color: #c7d2fe;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: .85rem;
    margin: 4px 4px 4px 0;
}

/* ---------- Risk badge ---------- */
.risk-badge {
    display: inline-block;
    border-radius: 8px;
    padding: 5px 14px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: .85rem;
    letter-spacing: .04em;
}
.risk-safe   { background: rgba(16,185,129,.18); color: #6ee7b7; border: 1px solid rgba(16,185,129,.3); }
.risk-medium { background: rgba(245,158,11,.15); color: #fcd34d; border: 1px solid rgba(245,158,11,.3); }
.risk-high   { background: rgba(239,68,68,.15);  color: #fca5a5; border: 1px solid rgba(239,68,68,.3); }

/* ---------- Chart background ---------- */
.stPlotlyChart, [data-testid="stImage"] { border-radius: 12px; }

/* ---------- Tab styling ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(255,255,255,.04) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,.07) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #6b7280 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
    padding: 8px 20px !important;
    transition: all .2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: #fff !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

/* ---------- Logout button (sidebar) ---------- */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: rgba(239,68,68,.15) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239,68,68,.3) !important;
    font-size: .85rem !important;
}

/* ---------- Dashboard logout buttons ---------- */
button[kind="secondary"]:has(div:contains("Logout")),
[data-testid="stButton"]:has(button[key="teacher_logout"]) > button,
[data-testid="stButton"]:has(button[key="student_logout"]) > button {
    background: rgba(239,68,68,.12) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239,68,68,.25) !important;
    font-size: .82rem !important;
    padding: 8px !important;
}

/* ---------- Role selector buttons styled as cards ---------- */
div[data-testid="stButton"]:has(> button[key="l_student_btn"]) > button,
div[data-testid="stButton"]:has(> button[key="l_teacher_btn"]) > button,
div[data-testid="stButton"]:has(> button[key="s_student_btn"]) > button,
div[data-testid="stButton"]:has(> button[key="s_teacher_btn"]) > button {
    background: rgba(255,255,255,.04) !important;
    border: 1.5px solid rgba(255,255,255,.12) !important;
    border-radius: 14px !important;
    padding: 20px 10px !important;
    height: 90px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: .8rem !important;
    font-weight: 700 !important;
    letter-spacing: .08em !important;
    color: #9ca3af !important;
    text-transform: uppercase !important;
    transition: all .25s cubic-bezier(.22,1,.36,1) !important;
    box-shadow: none !important;
}
div[data-testid="stButton"]:has(> button[key="l_student_btn"]) > button:hover,
div[data-testid="stButton"]:has(> button[key="l_teacher_btn"]) > button:hover,
div[data-testid="stButton"]:has(> button[key="s_student_btn"]) > button:hover,
div[data-testid="stButton"]:has(> button[key="s_teacher_btn"]) > button:hover {
    border-color: rgba(99,102,241,.6) !important;
    background: rgba(99,102,241,.08) !important;
    color: #a5b4fc !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,.2) !important;
}

/* ---------- Active role button (selected state) ---------- */
div[data-testid="stButton"]:has(> button[key="l_student_active"]) > button,
div[data-testid="stButton"]:has(> button[key="l_teacher_active"]) > button,
div[data-testid="stButton"]:has(> button[key="s_student_active"]) > button,
div[data-testid="stButton"]:has(> button[key="s_teacher_active"]) > button {
    background: rgba(99,102,241,.15) !important;
    border: 1.5px solid #6366f1 !important;
    color: #a5b4fc !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.15), 0 6px 20px rgba(99,102,241,.25) !important;
    height: 90px !important;
    border-radius: 14px !important;
    padding: 20px 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: .8rem !important;
    font-weight: 700 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
}

/* ---------- Prediction model cards ---------- */
.model-pred-row { display: flex; gap: 12px; flex-wrap: wrap; }
.model-card {
    flex: 1; min-width: 120px;
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.model-card.best {
    border-color: rgba(99,102,241,.5);
    background: rgba(99,102,241,.1);
}
.model-card .score {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #e8eaf0;
}
.model-card .name {
    font-size: .75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-top: 4px;
}
.best-tag {
    font-size: .65rem;
    background: #6366f1;
    color: #fff;
    border-radius: 4px;
    padding: 2px 6px;
    margin-top: 6px;
    display: inline-block;
}

/* =====================================================
   LOGIN PAGE — ADVANCED STYLES
   ===================================================== */

/* Animated particle canvas background (login only) */
#login-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
}

/* Entrance animation */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulseBorder {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
    50%       { box-shadow: 0 0 0 6px rgba(99,102,241,.12); }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-8px); }
}
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* Login form wrapper with entrance */
.login-anim {
    animation: fadeSlideUp .6s cubic-bezier(.22,1,.36,1) both;
}

/* Floating logo icon */
.login-icon {
    display: block;
    text-align: center;
    font-size: 3.2rem;
    animation: float 3.5s ease-in-out infinite;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 20px rgba(99,102,241,.5));
}

/* Logo text */
.login-brand {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #e8eaf0 20%, #a5b4fc 60%, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -.02em;
    line-height: 1;
}

/* Tagline with shimmer effect */
.login-tagline {
    text-align: center;
    font-size: .88rem;
    font-weight: 400;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: transparent;
    background: linear-gradient(90deg,
        #374151 0%, #6b7280 30%, #a5b4fc 50%, #6b7280 70%, #374151 100%);
    background-size: 400px 100%;
    -webkit-background-clip: text;
    animation: shimmer 3s linear infinite;
    margin-top: 8px;
}

/* Divider */
.login-divider {
    width: 56px; height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #8b5cf6, transparent);
    margin: 16px auto 28px;
    border-radius: 2px;
}

/* Role selector cards */
.role-cards {
    display: flex;
    gap: 10px;
    margin-bottom: 18px;
}
.role-card {
    flex: 1;
    background: rgba(255,255,255,.04);
    border: 1.5px solid rgba(255,255,255,.1);
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
    cursor: pointer;
    transition: all .25s cubic-bezier(.22,1,.36,1);
    position: relative;
    overflow: hidden;
}
.role-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(99,102,241,.15), rgba(139,92,246,.08));
    opacity: 0;
    transition: opacity .25s;
}
.role-card:hover::before { opacity: 1; }
.role-card:hover {
    border-color: rgba(99,102,241,.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99,102,241,.2);
}
.role-card.active {
    border-color: #6366f1;
    background: rgba(99,102,241,.12);
    box-shadow: 0 0 0 3px rgba(99,102,241,.15), 0 8px 24px rgba(99,102,241,.2);
    animation: pulseBorder 2.5s ease-in-out infinite;
}
.role-card.active::before { opacity: 1; }
.role-icon { font-size: 1.8rem; display: block; margin-bottom: 5px; }
.role-label {
    font-family: 'Syne', sans-serif;
    font-size: .82rem;
    font-weight: 700;
    color: #9ca3af;
    letter-spacing: .05em;
    text-transform: uppercase;
}
.role-card.active .role-label { color: #a5b4fc; }

/* Password strength bar */
.pw-strength-wrap { margin-top: 8px; margin-bottom: 4px; }
.pw-strength-bar {
    height: 4px;
    border-radius: 4px;
    transition: width .4s ease, background .4s ease;
    margin-bottom: 4px;
}
.pw-strength-label {
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .04em;
}

/* Checklist items */
.pw-checklist {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 8px;
}
.pw-check {
    font-size: .72rem;
    padding: 3px 9px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(255,255,255,.04);
    color: #6b7280;
    transition: all .2s;
}
.pw-check.ok {
    background: rgba(16,185,129,.12);
    border-color: rgba(16,185,129,.3);
    color: #6ee7b7;
}

/* Stats strip below form */
.login-stats {
    display: flex;
    justify-content: center;
    gap: 28px;
    margin-top: 28px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,.05);
}
.login-stat-item { text-align: center; }
.login-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e8eaf0, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.login-stat-lbl {
    font-size: .68rem;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-top: 2px;
}

/* Footer */
.login-footer {
    text-align: center;
    color: #1f2937;
    font-size: .72rem;
    margin-top: 20px;
    letter-spacing: .04em;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#13151d",
    "axes.facecolor":    "#13151d",
    "axes.edgecolor":    "#2d2f3a",
    "axes.labelcolor":   "#9ca3af",
    "xtick.color":       "#6b7280",
    "ytick.color":       "#6b7280",
    "text.color":        "#e8eaf0",
    "grid.color":        "#1e2030",
    "grid.alpha":        0.6,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
loaded      = joblib.load("best_model.pkl")
models      = loaded["models"]
best_model  = loaded["best_model"]
best_model_name = loaded["best_model_name"]
results     = loaded["results"]
features    = loaded["features"]
scaler      = loaded["scaler"]

# ─────────────────────────────────────────────
#  FILES
# ─────────────────────────────────────────────
USER_FILE    = "users.json"
HISTORY_FILE = "history.json"

for f in [USER_FILE, HISTORY_FILE]:
    if not os.path.exists(f):
        with open(f, "w") as file:
            json.dump({}, file)

def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def load_history():
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
def validate_email(email):
    return re.match(r"^[a-zA-Z0-9._%+-]+@gmail\.com$", email)

def validate_password(password):
    return (
        len(password) >= 8 and
        re.search(r"[A-Z]", password) and
        re.search(r"[0-9]", password) and
        re.search(r"[!@#$%^&*]", password)
    )

# ─────────────────────────────────────────────
#  SESSION
# ─────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ─────────────────────────────────────────────
#  LOGIN / SIGNUP
# ─────────────────────────────────────────────
if not st.session_state.logged_in:

    # ── Animated particle canvas (injected once) ──
    st.markdown("""
    <canvas id="login-bg"></canvas>
    <script>
    (function() {
        const canvas = document.getElementById('login-bg');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let W, H, particles = [];

        function resize() {
            W = canvas.width  = window.innerWidth;
            H = canvas.height = window.innerHeight;
        }
        resize();
        window.addEventListener('resize', resize);

        const COLORS = ['rgba(99,102,241,', 'rgba(139,92,246,', 'rgba(16,185,129,'];
        for (let i = 0; i < 55; i++) {
            particles.push({
                x:  Math.random() * 1400,
                y:  Math.random() * 900,
                r:  Math.random() * 1.8 + .4,
                dx: (Math.random() - .5) * .35,
                dy: (Math.random() - .5) * .35,
                c:  COLORS[Math.floor(Math.random() * COLORS.length)],
                a:  Math.random() * .5 + .1,
            });
        }

        function draw() {
            ctx.clearRect(0, 0, W, H);
            particles.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = p.c + p.a + ')';
                ctx.fill();
                p.x += p.dx; p.y += p.dy;
                if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
                if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
            });
            // draw faint connection lines
            for (let i = 0; i < particles.length; i++) {
                for (let j = i+1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < 120) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = 'rgba(99,102,241,' + (.12 * (1 - dist/120)) + ')';
                        ctx.lineWidth = .5;
                        ctx.stroke();
                    }
                }
            }
            requestAnimationFrame(draw);
        }
        draw();
    })();
    </script>
    """, unsafe_allow_html=True)

    # ── Init session state for role selection ──
    if "login_role" not in st.session_state:
        st.session_state.login_role = "Student"
    if "signup_role" not in st.session_state:
        st.session_state.signup_role = "Student"

    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        # ── Animated logo header ──
        st.markdown("""
        <div class="login-anim" style="text-align:center; margin-bottom:24px;">
            <span class="login-icon">🎓</span>
            <div class="login-brand">EduPredict</div>
            <div class="login-tagline">Predict &nbsp;·&nbsp; Improve &nbsp;·&nbsp; Succeed</div>
            <div class="login-divider"></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs ──
        tab_login, tab_signup = st.tabs(["🔑  Login", "✨  Sign Up"])

        # ══ LOGIN TAB ══
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)

            # Role selector — styled Streamlit buttons (no HTML overlay)
            st.markdown('<p style="color:#9ca3af;font-size:.82rem;margin-bottom:8px;letter-spacing:.06em;">SELECT YOUR ROLE</p>',
                        unsafe_allow_html=True)
            rc1, rc2 = st.columns(2)
            with rc1:
                is_active = st.session_state.login_role == "Student"
                key = "l_student_active" if is_active else "l_student_btn"
                if st.button("👨‍🎓\n\nStudent", key=key, use_container_width=True):
                    st.session_state.login_role = "Student"
                    st.rerun()
            with rc2:
                is_active = st.session_state.login_role == "Teacher"
                key = "l_teacher_active" if is_active else "l_teacher_btn"
                if st.button("👩‍🏫\n\nTeacher", key=key, use_container_width=True):
                    st.session_state.login_role = "Teacher"
                    st.rerun()

            st.markdown(
                f'<p style="color:#6366f1;font-size:.8rem;margin:10px 0 14px;font-weight:600;">'
                f'✓ Logging in as: {st.session_state.login_role}</p>',
                unsafe_allow_html=True,
            )

            email_l = st.text_input("📧 Gmail address", placeholder="you@gmail.com", key="email_l")
            pass_l  = st.text_input("🔒 Password", type="password", key="pass_l")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login  →", key="btn_login", use_container_width=True):
                users = load_users()
                if not validate_email(email_l):
                    st.error("⚠ Please enter a valid Gmail address.")
                elif email_l not in users:
                    st.error("⚠ Account not found. Please sign up first.")
                elif users[email_l]["password"] != pass_l:
                    st.error("⚠ Incorrect password.")
                else:
                    st.session_state.logged_in  = True
                    st.session_state.user_email = email_l
                    st.rerun()

            st.markdown(
                '<p style="text-align:center;color:#374151;font-size:.76rem;margin-top:14px;">'
                "Don't have an account? Switch to Sign Up ↑</p>",
                unsafe_allow_html=True,
            )

        # ══ SIGNUP TAB ══
        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)

            # Role selector — styled Streamlit buttons (no HTML overlay)
            st.markdown('<p style="color:#9ca3af;font-size:.82rem;margin-bottom:8px;letter-spacing:.06em;">SELECT YOUR ROLE</p>',
                        unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                is_active = st.session_state.signup_role == "Student"
                key = "s_student_active" if is_active else "s_student_btn"
                if st.button("👨‍🎓\n\nStudent", key=key, use_container_width=True):
                    st.session_state.signup_role = "Student"
                    st.rerun()
            with sc2:
                is_active = st.session_state.signup_role == "Teacher"
                key = "s_teacher_active" if is_active else "s_teacher_btn"
                if st.button("👩‍🏫\n\nTeacher", key=key, use_container_width=True):
                    st.session_state.signup_role = "Teacher"
                    st.rerun()

            st.markdown(
                f'<p style="color:#6366f1;font-size:.8rem;margin:10px 0 14px;font-weight:600;">'
                f'✓ Signing up as: {st.session_state.signup_role}</p>',
                unsafe_allow_html=True,
            )

            email_s = st.text_input("📧 Gmail address", placeholder="you@gmail.com", key="email_s")
            pass_s  = st.text_input("🔒 Password", type="password", key="pass_s")

            # ── Live password strength indicator ──
            if pass_s:
                has_len   = len(pass_s) >= 8
                has_upper = bool(re.search(r"[A-Z]", pass_s))
                has_num   = bool(re.search(r"[0-9]", pass_s))
                has_spec  = bool(re.search(r"[!@#$%^&*]", pass_s))
                score     = sum([has_len, has_upper, has_num, has_spec])

                bar_configs = {
                    1: ("#ef4444", "25%",  "Weak"),
                    2: ("#f59e0b", "50%",  "Fair"),
                    3: ("#6366f1", "75%",  "Good"),
                    4: ("#10b981", "100%", "Strong ✓"),
                }
                color, width, label = bar_configs.get(score, ("#ef4444", "10%", "Too short"))

                def check_html(ok, text):
                    cls = "ok" if ok else ""
                    return f'<span class="pw-check {cls}">{text}</span>'

                st.markdown(f"""
                <div class="pw-strength-wrap">
                    <div style="background:rgba(255,255,255,.06);border-radius:4px;overflow:hidden;">
                        <div class="pw-strength-bar"
                             style="width:{width};background:{color};"></div>
                    </div>
                    <span class="pw-strength-label" style="color:{color};">{label}</span>
                    <div class="pw-checklist">
                        {check_html(has_len,   "8+ chars")}
                        {check_html(has_upper, "Uppercase")}
                        {check_html(has_num,   "Number")}
                        {check_html(has_spec,  "Special (!@#...)")}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<p style="color:#374151;font-size:.76rem;margin-top:4px;">'
                    'Min 8 chars · uppercase · number · special (!@#$%^&*)</p>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account  →", key="btn_signup", use_container_width=True):
                users = load_users()
                if not validate_email(email_s):
                    st.error("⚠ Please enter a valid Gmail address.")
                elif not validate_password(pass_s):
                    st.error("⚠ Weak password — need 8+ chars, uppercase, number & special char.")
                elif email_s in users:
                    st.error("⚠ This email is already registered. Try logging in.")
                else:
                    users[email_s] = {"password": pass_s, "role": st.session_state.signup_role}
                    save_users(users)
                    st.session_state.logged_in  = True
                    st.session_state.user_email = email_s
                    st.rerun()

            st.markdown(
                '<p style="text-align:center;color:#374151;font-size:.76rem;margin-top:14px;">'
                "Already have an account? Switch to Login ↑</p>",
                unsafe_allow_html=True,
            )

        # ── Stats strip ──
        users_count = len(load_users())
        st.markdown(f"""
        <div class="login-stats">
            <div class="login-stat-item">
                <div class="login-stat-val">{users_count}</div>
                <div class="login-stat-lbl">Users</div>
            </div>
            <div class="login-stat-item">
                <div class="login-stat-val">3</div>
                <div class="login-stat-lbl">ML Models</div>
            </div>
            <div class="login-stat-item">
                <div class="login-stat-val">6</div>
                <div class="login-stat-lbl">Features</div>
            </div>
        </div>
        <div class="login-footer">EduPredict · Student Performance Analytics</div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN APP (logged in)
# ─────────────────────────────────────────────
else:
    users      = load_users()
    history    = load_history()
    user_email = st.session_state.user_email
    role       = users[user_email]["role"]

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            f'<div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;'
            f'margin-bottom:4px;">🎓 EduPredict</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"<small style='color:#6b7280'>{user_email}</small>", unsafe_allow_html=True)
        badge_cls = "risk-safe" if role == "Teacher" else "risk-medium"
        st.markdown(
            f'<span class="risk-badge {badge_cls}" style="margin:12px 0;display:inline-block">'
            f'{"👩‍🏫" if role == "Teacher" else "👨‍🎓"} {role}</span>',
            unsafe_allow_html=True,
        )
        st.divider()
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # ══════════════════════════════════════════
    #  TEACHER DASHBOARD
    # ══════════════════════════════════════════
    if role == "Teacher":

        _t1, _t2 = st.columns([5, 1])
        with _t1:
            st.markdown('<div class="page-title">Teacher Dashboard</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-sub">Predict and track student exam performance</div>', unsafe_allow_html=True)
        with _t2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⇠ Logout", key="teacher_logout", use_container_width=True):
                st.session_state.logged_in = False
                st.rerun()

        # ── Input Form ──
        st.markdown('<div class="section-header">🔍 Student Prediction</div>', unsafe_allow_html=True)

        student_email = st.text_input(
            "Student Gmail (to save record)",
            placeholder="student@gmail.com",
        )
        if student_email and not validate_email(student_email):
            st.error("Enter a valid student Gmail address.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            study_hours     = st.slider("📚 Study Hours / Day",  0, 8,   5)
            social_media    = st.slider("📱 Social Media Hours", 0, 10,  2)
            part_time_job   = st.selectbox("💼 Part-Time Job",   ["No", "Yes"])
        with col2:
            attendance      = st.slider("🏫 Attendance (%)",     0, 100, 60)
            sleep_hours     = st.slider("😴 Sleep Hours",        4, 10,  6)
            extracurricular = st.selectbox("⚽ Extracurricular", ["No", "Yes"])

        predict_clicked = st.button("⚡ Predict Score")

        # ── Prediction results (only when button clicked) ──
        if predict_clicked:
            ptj   = 1 if part_time_job  == "Yes" else 0
            extra = 1 if extracurricular == "Yes" else 0

            input_dict = {
                "study_hours_per_day":          study_hours,
                "social_media_hours":           social_media,
                "part_time_job":                ptj,
                "attendance_percentage":        attendance,
                "sleep_hours":                  sleep_hours,
                "extracurricular_participation": extra,
            }

            input_df   = pd.DataFrame([[input_dict[f] for f in features]], columns=features)
            input_data = scaler.transform(input_df)

            # All model predictions
            all_predictions = {}
            for name, m in models.items():
                pred = m.predict(input_data)[0]
                pred = max(0, min(100, pred))
                all_predictions[name] = pred

            prediction = all_predictions[best_model_name]

            # Risk label
            if prediction < 50:
                risk = "High Risk"
            elif prediction < 70:
                risk = "Medium Risk"
            else:
                risk = "Safe"

            # ── Score & Risk summary ──
            st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
            risk_cls = {"Safe": "risk-safe", "Medium Risk": "risk-medium", "High Risk": "risk-high"}[risk]

            r1, r2, r3 = st.columns(3)
            with r1:
                colour = "green" if risk == "Safe" else ("amber" if risk == "Medium Risk" else "red")
                st.markdown(
                    f'<div class="metric-tile {colour}">'
                    f'<div class="val">{prediction:.1f}</div>'
                    f'<div class="lbl">Predicted Score</div></div>',
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="val" style="font-size:1.3rem">'
                    f'<span class="risk-badge {risk_cls}">{risk}</span></div>'
                    f'<div class="lbl" style="margin-top:10px">Risk Level</div></div>',
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="val" style="font-size:1.1rem">{best_model_name}</div>'
                    f'<div class="lbl">Best Model Used</div></div>',
                    unsafe_allow_html=True,
                )

            # ── All model predictions ──
            st.markdown('<div class="section-header">🤖 All Model Predictions</div>', unsafe_allow_html=True)
            cards_html = '<div class="model-pred-row">'
            for name, pred in all_predictions.items():
                is_best = name == best_model_name
                cards_html += (
                    f'<div class="model-card {"best" if is_best else ""}">'
                    f'<div class="score">{pred:.1f}</div>'
                    f'<div class="name">{name}</div>'
                    f'{"<div class=\'best-tag\'>★ Best</div>" if is_best else ""}'
                    f'</div>'
                )
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            # ── Model comparison table (always visible, styled) ──
            st.markdown('<div class="section-header">📋 Model Comparison</div>', unsafe_allow_html=True)
            st.dataframe(
                loaded["results"].style.highlight_min(subset=["rmse"], color="#1a2a1a")
                                       .highlight_max(subset=["r2"],   color="#1a2a1a")
                                       .format({"rmse": "{:.4f}", "r2": "{:.4f}"}),
                use_container_width=True,
            )
            st.info(
                f"**{best_model_name}** selected as best model — "
                "it achieved the lowest RMSE (highlighted), indicating the highest prediction accuracy."
            )

            # ── Bar chart of model predictions ──
            fig, ax = plt.subplots(figsize=(7, 3))
            bars = ax.bar(
                all_predictions.keys(),
                all_predictions.values(),
                color=["#6366f1" if n == best_model_name else "#374151" for n in all_predictions],
                edgecolor="none",
                width=0.5,
            )
            ax.set_ylabel("Predicted Score")
            ax.set_title("Model Prediction Comparison", color="#e8eaf0", fontweight="bold")
            ax.set_ylim(0, 100)
            ax.axhline(prediction, color="#10b981", linestyle="--", linewidth=1, alpha=0.6)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Suggestions ──
            st.markdown('<div class="section-header">💡 Suggestions</div>', unsafe_allow_html=True)
            suggestions = []
            if prediction < 60:
                suggestions.append("Focus on studies more consistently")
            if study_hours < 4:
                suggestions.append("Increase study hours to at least 4 hours / day")
            if social_media > 5:
                suggestions.append("Reduce social media usage")
            if attendance < 50:
                suggestions.append("Improve class attendance urgently")
            if sleep_hours < 6:
                suggestions.append("Sleep at least 6–7 hours daily")
            if not suggestions:
                if risk == "High Risk":
                    suggestions.append("Immediate improvement needed in study habits")
                elif risk == "Medium Risk":
                    suggestions.append("Try improving consistency in daily habits")
                else:
                    suggestions.append("Great job! Keep maintaining your routine! 🎉")

            pills = "".join(f'<span class="suggestion-pill">✔ {s}</span>' for s in suggestions)
            st.markdown(f'<div style="margin-top:8px">{pills}</div>', unsafe_allow_html=True)

            # ── Save to history ──
            if student_email:
                if student_email not in history:
                    history[student_email] = []
                history[student_email].append({
                    "score":       float(prediction),
                    "risk":        risk,
                    "study_hours": study_hours,
                    "social_media": social_media,
                    "attendance":  attendance,
                    "sleep_hours": sleep_hours,
                    "suggestions": suggestions,
                })
                save_history(history)
                st.success(f"✅ Record saved for **{student_email}**")

        # ── Class Overview (always visible to teacher) ──
        st.markdown('<div class="section-header">📈 Class Performance Overview</div>', unsafe_allow_html=True)

        if history:
            # ── Build one row per STUDENT (average score → overall risk) ──
            student_summary = {}
            for email, records in history.items():
                if not records:
                    continue
                scores_list   = [r["score"] for r in records]
                avg_score     = np.mean(scores_list)
                latest_record = records[-1]

                if avg_score < 50:
                    overall_risk = "High Risk"
                elif avg_score < 70:
                    overall_risk = "Medium Risk"
                else:
                    overall_risk = "Safe"

                # Count how many times each risk appeared across all records
                risk_counts_student = {"Safe": 0, "Medium Risk": 0, "High Risk": 0}
                for r in records:
                    risk_counts_student[r["risk"]] = risk_counts_student.get(r["risk"], 0) + 1

                student_summary[email] = {
                    "avg_score":    round(avg_score, 1),
                    "latest_score": round(scores_list[-1], 1),
                    "attempts":     len(records),
                    "overall_risk": overall_risk,
                    "safe_count":   risk_counts_student["Safe"],
                    "medium_count": risk_counts_student["Medium Risk"],
                    "high_count":   risk_counts_student["High Risk"],
                    "study_hours":  latest_record.get("study_hours", "—"),
                    "attendance":   latest_record.get("attendance",  "—"),
                    "sleep_hours":  latest_record.get("sleep_hours", "—"),
                }

            safe_students   = {e: d for e, d in student_summary.items() if d["overall_risk"] == "Safe"}
            medium_students = {e: d for e, d in student_summary.items() if d["overall_risk"] == "Medium Risk"}
            high_students   = {e: d for e, d in student_summary.items() if d["overall_risk"] == "High Risk"}
            all_avgs        = [d["avg_score"] for d in student_summary.values()]

            # ── Summary tiles (per STUDENT, not per prediction) ──
            t1, t2, t3, t4 = st.columns(4)
            tiles = [
                (t1, f"{np.mean(all_avgs):.1f}", "Class Avg Score",        ""),
                (t2, str(len(safe_students)),     "Safe Students",          "green"),
                (t3, str(len(medium_students)),   "Medium Risk Students",   "amber"),
                (t4, str(len(high_students)),     "High Risk Students",     "red"),
            ]
            for col, val, lbl, cls in tiles:
                with col:
                    st.markdown(
                        f'<div class="metric-tile {cls}">'
                        f'<div class="val">{val}</div>'
                        f'<div class="lbl">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Per-category expandable student lists ──
            def student_detail_table(students_dict, risk_label, border_color):
                if not students_dict:
                    st.caption(f"No {risk_label} students.")
                    return
                rows = []
                for email, d in students_dict.items():
                    rows.append({
                        "Student Email":    email,
                        "Avg Score":        d["avg_score"],
                        "Latest Score":     d["latest_score"],
                        "Attempts":         d["attempts"],
                        "✅ Safe Pred.":    d["safe_count"],
                        "🟡 Medium Pred.":  d["medium_count"],
                        "🔴 High Pred.":    d["high_count"],
                        "Study Hrs (last)": d["study_hours"],
                        "Attendance (last)":d["attendance"],
                        "Sleep Hrs (last)": d["sleep_hours"],
                    })
                df_display = pd.DataFrame(rows)
                st.dataframe(df_display, use_container_width=True, hide_index=True)

            exp1, exp2, exp3 = st.columns(3)

            with exp1:
                with st.expander(f"✅ Safe Students ({len(safe_students)})", expanded=False):
                    student_detail_table(safe_students, "Safe", "#10b981")

            with exp2:
                with st.expander(f"🟡 Medium Risk Students ({len(medium_students)})", expanded=False):
                    student_detail_table(medium_students, "Medium Risk", "#f59e0b")

            with exp3:
                with st.expander(f"🔴 High Risk Students ({len(high_students)})", expanded=False):
                    student_detail_table(high_students, "High Risk", "#ef4444")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Two charts ──
            # Collect per-prediction data for scatter (study hours vs score)
            x_vals, y_vals, colors_scatter = [], [], []
            trend_scores = []
            for email, records in history.items():
                for r in records:
                    trend_scores.append(r["score"])
                    x_vals.append(r.get("study_hours", 0))
                    y_vals.append(r["score"])
                    overall = student_summary[email]["overall_risk"]
                    if overall == "High Risk":
                        colors_scatter.append("#ef4444")
                    elif overall == "Medium Risk":
                        colors_scatter.append("#f59e0b")
                    else:
                        colors_scatter.append("#10b981")

            ch1, ch2 = st.columns(2)
            with ch1:
                fig1, ax1 = plt.subplots(figsize=(5, 3))
                ax1.plot(trend_scores, color="#6366f1", linewidth=2, marker="o",
                         markersize=4, markerfacecolor="#a5b4fc")
                ax1.fill_between(range(len(trend_scores)), trend_scores,
                                 alpha=0.12, color="#6366f1")
                ax1.axhline(70, color="#10b981", linestyle="--", linewidth=1,
                            alpha=0.5, label="Target (70)")
                ax1.set_title("Class Performance Trend", color="#e8eaf0", fontweight="bold")
                ax1.set_xlabel("Predictions (chronological)")
                ax1.set_ylabel("Score")
                ax1.legend(facecolor="#13151d", edgecolor="#2d2f3a", labelcolor="#9ca3af")
                fig1.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

            with ch2:
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.scatter(x_vals, y_vals, c=colors_scatter, s=80, alpha=0.75, edgecolors="none")
                ax2.set_title("Risk Analysis: Study Hours vs Score",
                              color="#e8eaf0", fontweight="bold")
                ax2.set_xlabel("Study Hours / Day")
                ax2.set_ylabel("Predicted Score")
                patches = [
                    mpatches.Patch(color="#10b981", label="Safe (overall)"),
                    mpatches.Patch(color="#f59e0b", label="Medium Risk (overall)"),
                    mpatches.Patch(color="#ef4444", label="High Risk (overall)"),
                ]
                ax2.legend(handles=patches, facecolor="#13151d", edgecolor="#2d2f3a",
                           labelcolor="#9ca3af")
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

        else:
            st.info("No student data recorded yet. Use the prediction form above to add records.")

    # ══════════════════════════════════════════
    #  STUDENT DASHBOARD
    # ══════════════════════════════════════════
    else:
        _s1, _s2 = st.columns([5, 1])
        with _s1:
            st.markdown('<div class="page-title">My Dashboard</div>', unsafe_allow_html=True)
            st.markdown('<div class="page-sub">Track your predicted performance and habits</div>', unsafe_allow_html=True)
        with _s2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⇠ Logout", key="student_logout", use_container_width=True):
                st.session_state.logged_in = False
                st.rerun()

        user_data = history.get(user_email, [])

        if not user_data:
            st.markdown(
                '<div class="card" style="text-align:center;padding:48px">'
                '<div style="font-size:3rem">📭</div>'
                '<div style="font-family:Syne,sans-serif;font-size:1.2rem;margin:12px 0">No records yet</div>'
                '<div style="color:#6b7280">Your teacher hasn\'t made a prediction for you yet. Check back later.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            latest = user_data[-1]
            score  = latest["score"]
            risk   = latest["risk"]

            # Risk colour
            risk_cls = {"Safe": "risk-safe", "Medium Risk": "risk-medium", "High Risk": "risk-high"}[risk]
            score_color = "green" if risk == "Safe" else ("amber" if risk == "Medium Risk" else "red")

            # ── Summary cards ──
            st.markdown('<div class="section-header">📊 Latest Result</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f'<div class="metric-tile {score_color}">'
                    f'<div class="val">{score:.1f}</div>'
                    f'<div class="lbl">Latest Score</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                avg = np.mean([x["score"] for x in user_data])
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="val">{avg:.1f}</div>'
                    f'<div class="lbl">Average Score</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="val">{len(user_data)}</div>'
                    f'<div class="lbl">Total Attempts</div></div>',
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div style="margin-top:6px"><span class="risk-badge {risk_cls}">{risk}</span></div>'
                    f'<div class="lbl" style="margin-top:10px">Risk Level</div></div>',
                    unsafe_allow_html=True,
                )

            # ── Habit snapshot ──
            st.markdown('<div class="section-header">📋 Latest Habit Snapshot</div>', unsafe_allow_html=True)
            h1, h2, h3, h4 = st.columns(4)
            habit_tiles = [
                (h1, f'{latest.get("study_hours","—")}h',   "Study Hours/Day"),
                (h2, f'{latest.get("social_media","—")}h',  "Social Media"),
                (h3, f'{latest.get("attendance","—")}%',     "Attendance"),
                (h4, f'{latest.get("sleep_hours","—")}h',   "Sleep Hours"),
            ]
            for col, val, lbl in habit_tiles:
                with col:
                    st.markdown(
                        f'<div class="metric-tile">'
                        f'<div class="val" style="font-size:1.5rem">{val}</div>'
                        f'<div class="lbl">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            # ── Suggestions ──
            st.markdown('<div class="section-header">💡 Suggestions</div>', unsafe_allow_html=True)
            suggestions = latest.get("suggestions", [])
            if not suggestions:
                st.markdown('<span class="suggestion-pill">✔ Keep up the great work! 🎉</span>',
                            unsafe_allow_html=True)
            else:
                pills = "".join(f'<span class="suggestion-pill">✔ {s}</span>' for s in suggestions)
                st.markdown(pills, unsafe_allow_html=True)

            # ── Progress chart ──
            st.markdown('<div class="section-header">📈 Your Progress</div>', unsafe_allow_html=True)
            scores = [x["score"] for x in user_data]

            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(scores, color="#6366f1", linewidth=2.5, marker="o",
                    markersize=6, markerfacecolor="#a5b4fc", zorder=3)
            ax.fill_between(range(len(scores)), scores, alpha=0.1, color="#6366f1")
            ax.axhline(70, color="#10b981", linestyle="--", linewidth=1, alpha=0.5, label="Target (70)")
            ax.set_title("Performance Progress Over Attempts", color="#e8eaf0", fontweight="bold")
            ax.set_xlabel("Attempt #")
            ax.set_ylabel("Predicted Score")
            ax.set_ylim(0, 105)
            ax.legend(facecolor="#13151d", edgecolor="#2d2f3a", labelcolor="#9ca3af")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)