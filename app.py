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
    initial_sidebar_state="expanded",
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
    background: #0d0f14 !important;
    border-right: 1px solid rgba(255,255,255,.07) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] * { color: #c8cad4 !important; }
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* Sidebar nav item styling */
.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 10px;
    margin: 3px 0;
    cursor: pointer;
    font-family: 'DM Sans', sans-serif;
    font-size: .9rem;
    font-weight: 500;
    color: #6b7280;
    transition: all .2s;
    border: 1px solid transparent;
}
.nav-item:hover {
    background: rgba(99,102,241,.1);
    color: #a5b4fc;
    border-color: rgba(99,102,241,.2);
}
.nav-item.active {
    background: rgba(99,102,241,.15);
    color: #a5b4fc;
    border-color: rgba(99,102,241,.3);
    font-weight: 600;
}
.nav-section-label {
    font-size: .68rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #374151;
    padding: 16px 16px 6px;
    font-weight: 600;
}
.sidebar-brand {
    padding: 24px 16px 20px;
    border-bottom: 1px solid rgba(255,255,255,.06);
    margin-bottom: 8px;
}
.sidebar-brand-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e8eaf0 30%, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-user-email {
    font-size: .75rem;
    color: #4b5563;
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sidebar-role-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .04em;
}
.sidebar-divider {
    height: 1px;
    background: rgba(255,255,255,.06);
    margin: 10px 16px;
}
.sidebar-logout-wrap {
    padding: 10px 16px 20px;
    border-top: 1px solid rgba(255,255,255,.06);
    margin-top: auto;
}

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

/* ---------- Logout / secondary buttons ---------- */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: rgba(239,68,68,.12) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239,68,68,.25) !important;
    font-size: .85rem !important;
}
div[data-testid="stButton"]:has(> button[key="teacher_logout"]) > button,
div[data-testid="stButton"]:has(> button[key="student_logout"]) > button {
    background: rgba(239,68,68,.12) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239,68,68,.25) !important;
    font-size: .82rem !important;
    padding: 8px !important;
}

/* ---------- Role selector buttons ---------- */
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
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* ✅ ADD HERE */
body {
    animation: fadeUp .8s ease;
}

body::after {
    content: "";
    position: fixed;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,.15), transparent 70%);
    pointer-events: none;
    transform: translate(-50%, -50%);
    top: var(--y, 50%);
    left: var(--x, 50%);
    z-index: 0;
}
/* ─── LANDING PAGE ─── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(32px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes floatY {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-10px); }
}
@keyframes glow {
    0%,100% { opacity: .6; }
    50%      { opacity: 1; }
}
@keyframes shimmerText {
    0%   { background-position: -600px 0; }
    100% { background-position: 600px 0; }
}
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

.landing-hero {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 60px 20px 40px;
    position: relative;
}
.landing-hero::before {
    content: "";
    position: absolute;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(99,102,241,0.25), transparent 70%);
    top: -100px;
    left: -100px;
    z-index: 0;
    filter: blur(80px);
}
.landing-icon-wrap {
    position: relative;
    display: inline-block;
    margin-bottom: 28px;
    animation: floatY 4s ease-in-out infinite;
}
.landing-icon {
    font-size: 5rem;
    display: block;
    filter: drop-shadow(0 0 30px rgba(99,102,241,.6));
}
.landing-icon-ring {
    position: absolute;
    inset: -16px;
    border-radius: 50%;
    border: 1.5px solid rgba(99,102,241,.25);
    animation: spin-slow 8s linear infinite, glow 3s ease-in-out infinite;
}
.landing-icon-ring2 {
    position: absolute;
    inset: -30px;
    border-radius: 50%;
    border: 1px solid rgba(139,92,246,.12);
    animation: spin-slow 14s linear infinite reverse;
}

.landing-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,.12);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: .78rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: 20px;
    animation: fadeUp .6s .1s both;
}
.landing-badge-dot {
    width: 6px; height: 6px;
    background: #6366f1;
    border-radius: 50%;
    animation: glow 1.5s ease-in-out infinite;
}

.landing-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 800;
    letter-spacing: -.03em;

    background: linear-gradient(90deg, #ffffff, #a5b4fc, #6366f1, #a5b4fc);
    background-size: 300% 100%;
    animation: shimmerText 6s linear infinite;

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    text-shadow: 0 0 30px rgba(99,102,241,0.3);
}

.landing-sub {
    font-size: 1.15rem;
    color: #9ca3af;
    max-width: 600px;
    line-height: 1.8;
    font-weight: 300;
    margin-bottom: 44px;
    animation: fadeUp .6s .3s both;
}

.landing-cta-row {
    display: flex;
    gap: 14px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 64px;
    animation: fadeUp .6s .4s both;
}

.landing-features {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    justify-content: center;
    max-width: 860px;
    animation: fadeUp .6s .5s both;
}
.landing-feature-card {
    flex: 1;
    min-width: 220px;
    max-width: 260px;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: left;
    transition: all .3s;
}
.landing-feature-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 
        0 20px 50px rgba(99,102,241,.25),
        inset 0 0 20px rgba(99,102,241,.05);
}
.lfc-icon { font-size: 1.8rem; margin-bottom: 12px; }
.lfc-title {
    font-family: 'Syne', sans-serif;
    font-size: .95rem;
    font-weight: 700;
    color: #e8eaf0;
    margin-bottom: 6px;
}
.lfc-desc { font-size: .82rem; color: #4b5563; line-height: 1.6; }

.landing-stats-row {
    display: flex;
    gap: 36px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 56px;
    padding-top: 36px;
    border-top: 1px solid rgba(255,255,255,.05);
    animation: fadeUp .6s .6s both;
}
.landing-stat {
    text-align: center;
}
.landing-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e8eaf0, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.landing-stat-lbl {
    font-size: .72rem;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: 4px;
}

/* ─── LOGIN PAGE ─── */
@keyframes pulseGlow {
    0%   { box-shadow: 0 0 0 0 rgba(99,102,241,0.6); }
    70%  { box-shadow: 0 0 0 12px rgba(99,102,241,0); }
    100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-8px); }
}

.login-anim {
    animation: fadeUp .6s cubic-bezier(.22,1,.36,1) both;
}
.login-icon {
    display: block;
    text-align: center;
    font-size: 3.2rem;
    animation: float 3.5s ease-in-out infinite;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 20px rgba(99,102,241,.5));
}
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
.login-divider {
    width: 56px; height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #8b5cf6, transparent);
    margin: 16px auto 28px;
    border-radius: 2px;
}
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
.login-footer {
    text-align: center;
    color: #1f2937;
    font-size: .72rem;
    margin-top: 20px;
    letter-spacing: .04em;
}

/* ─── CTA buttons on landing ─── */
div[data-testid="stButton"]:has(> button[key="landing_get_started"]) > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    font-size: 1rem !important;
    width: auto !important;
}
div[data-testid="stButton"]:has(> button[key="landing_get_started"]) > button {
    box-shadow: 0 0 0 rgba(99,102,241,0.6);
    animation: pulseGlow 2s infinite;
}
div[data-testid="stButton"]:has(> button[key="landing_learn_more"]) > button {
    background: rgba(255,255,255,.05) !important;
    color: #9ca3af !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    font-size: 1rem !important;
    width: auto !important;
}
div[data-testid="stButton"]:has(> button[key="landing_learn_more"]) > button:hover {
    border-color: rgba(99,102,241,.4) !important;
    color: #a5b4fc !important;
    box-shadow: none !important;
}

/* Sidebar nav buttons */
div[data-testid="stButton"]:has(> button[key^="nav_"]) > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    color: #6b7280 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .9rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 10px 14px !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
    transform: none !important;
}
div[data-testid="stButton"]:has(> button[key^="nav_"]) > button:hover {
    background: rgba(99,102,241,.1) !important;
    border-color: rgba(99,102,241,.2) !important;
    color: #a5b4fc !important;
    transform: none !important;
    box-shadow: none !important;
}
div[data-testid="stButton"]:has(> button[key^="nav_active_"]) > button {
    background: rgba(99,102,241,.15) !important;
    border: 1px solid rgba(99,102,241,.3) !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .9rem !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 10px 14px !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
    transform: none !important;
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
loaded           = joblib.load("best_model.pkl")
models           = loaded["models"]
best_model       = loaded["best_model"]
best_model_name  = loaded["best_model_name"]
results          = loaded["results"]
features         = loaded["features"]
scaler           = loaded["scaler"]

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
#  SESSION DEFAULTS
# ─────────────────────────────────────────────
for key, default in [
    ("logged_in", False),
    ("show_auth", False),
    ("login_role", "Student"),
    ("signup_role", "Student"),
    ("teacher_section", "Predict"),
    ("student_section", "Overview"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════
if not st.session_state.logged_in and not st.session_state.show_auth:

    # Particle background
    st.markdown("""
    <canvas id="login-bg" style="position:fixed;inset:0;z-index:0;pointer-events:none;"></canvas>
    <script>
    (function() {
        const canvas = document.getElementById('login-bg');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let W, H, particles = [];
        function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
        resize(); window.addEventListener('resize', resize);
        const COLORS = ['rgba(99,102,241,','rgba(139,92,246,','rgba(16,185,129,'];
        for (let i = 0; i < 55; i++) {
            particles.push({ x:Math.random()*1400, y:Math.random()*900, r:Math.random()*1.8+.4,
                dx:(Math.random()-.5)*.35, dy:(Math.random()-.5)*.35,
                c:COLORS[Math.floor(Math.random()*COLORS.length)], a:Math.random()*.5+.1 });
        }
        function draw() {
            ctx.clearRect(0,0,W,H);
            particles.forEach(p => {
                ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
                ctx.fillStyle=p.c+p.a+')'; ctx.fill();
                p.x+=p.dx; p.y+=p.dy;
                if(p.x<0)p.x=W; if(p.x>W)p.x=0;
                if(p.y<0)p.y=H; if(p.y>H)p.y=0;
            });
            for(let i=0;i<particles.length;i++) for(let j=i+1;j<particles.length;j++) {
                const dx=particles[i].x-particles[j].x, dy=particles[i].y-particles[j].y;
                const dist=Math.sqrt(dx*dx+dy*dy);
                if(dist<120) {
                    ctx.beginPath(); ctx.moveTo(particles[i].x,particles[i].y);
                    ctx.lineTo(particles[j].x,particles[j].y);
                    ctx.strokeStyle='rgba(99,102,241,'+(.12*(1-dist/120))+')';
                    ctx.lineWidth=.5; ctx.stroke();
                }
            }
            requestAnimationFrame(draw);
        }
        draw();
    })();
    </script>
    """, unsafe_allow_html=True)

    users_count = len(load_users())

    # Hero
    st.markdown(
        '<h1 style="color:white;">EduPredict</h1>'
        '<p style="color:gray;">A smart system to predict student performance</p>',
        unsafe_allow_html=True
    )

    # CTA buttons
    _, c1, _ = st.columns([2, 1, 2])
    with c1:
        if st.button("🚀  Get Started", key="landing_get_started", use_container_width=True):
            st.session_state.show_auth = True
            st.rerun()

    # Feature cards
    st.markdown("""
    <div style="display:flex;gap:18px;flex-wrap:wrap;justify-content:center;max-width:900px;margin:0 auto 48px;animation:fadeUp .6s .5s both;">
        <div class="landing-feature-card">
            <div class="lfc-icon">🤖</div>
            <div class="lfc-title">Multi-Model ML</div>
            <div class="lfc-desc">Three models compete — Linear Regression, Decision Tree, and Random Forest — and the best one wins.</div>
        </div>
        <div class="landing-feature-card">
            <div class="lfc-icon">📊</div>
            <div class="lfc-title">Real-Time Analytics</div>
            <div class="lfc-desc">Teachers get a live class overview with risk distributions, trend charts, and per-student breakdowns.</div>
        </div>
        <div class="landing-feature-card">
            <div class="lfc-icon">💡</div>
            <div class="lfc-title">Smart Suggestions</div>
            <div class="lfc-desc">Personalised, habit-based recommendations generated instantly for every prediction.</div>
        </div>
        <div class="landing-feature-card">
            <div class="lfc-icon">🔒</div>
            <div class="lfc-title">Role-Based Access</div>
            <div class="lfc-desc">Separate Teacher and Student dashboards — each seeing only what's relevant to them.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats strip
    st.markdown(f"""
    <div class="landing-stats-row">
        <div class="landing-stat">
            <div class="landing-stat-val">{users_count}</div>
            <div class="landing-stat-lbl">Registered Users</div>
        </div>
        <div class="landing-stat">
            <div class="landing-stat-val">3</div>
            <div class="landing-stat-lbl">ML Models</div>
        </div>
        <div class="landing-stat">
            <div class="landing-stat-val">6</div>
            <div class="landing-stat-lbl">Input Features</div>
        </div>
        <div class="landing-stat">
            <div class="landing-stat-val">2</div>
            <div class="landing-stat-lbl">Role Dashboards</div>
        </div>
    </div>
    <div style="text-align:center;color:#1f2937;font-size:.72rem;margin-top:24px;letter-spacing:.04em;">
        EduPredict · Student Performance Analytics
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════
#  LOGIN / SIGNUP PAGE
# ══════════════════════════════════════════════
if not st.session_state.logged_in and st.session_state.show_auth:

    # Particle background
    st.markdown("""
    <canvas id="login-bg" style="position:fixed;inset:0;z-index:0;pointer-events:none;"></canvas>
    <script>
    (function() {
        const canvas = document.getElementById('login-bg');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let W, H, particles = [];
        function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
        resize(); window.addEventListener('resize', resize);
        const COLORS = ['rgba(99,102,241,','rgba(139,92,246,','rgba(16,185,129,'];
        for (let i = 0; i < 55; i++) {
            particles.push({ x:Math.random()*1400, y:Math.random()*900, r:Math.random()*1.8+.4,
                dx:(Math.random()-.5)*.35, dy:(Math.random()-.5)*.35,
                c:COLORS[Math.floor(Math.random()*COLORS.length)], a:Math.random()*.5+.1 });
        }
        function draw() {
            ctx.clearRect(0,0,W,H);
            particles.forEach(p => {
                ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
                ctx.fillStyle=p.c+p.a+')'; ctx.fill();
                p.x+=p.dx; p.y+=p.dy;
                if(p.x<0)p.x=W; if(p.x>W)p.x=0;
                if(p.y<0)p.y=H; if(p.y>H)p.y=0;
            });
            for(let i=0;i<particles.length;i++) for(let j=i+1;j<particles.length;j++) {
                const dx=particles[i].x-particles[j].x, dy=particles[i].y-particles[j].y;
                const dist=Math.sqrt(dx*dx+dy*dy);
                if(dist<120) {
                    ctx.beginPath(); ctx.moveTo(particles[i].x,particles[i].y);
                    ctx.lineTo(particles[j].x,particles[j].y);
                    ctx.strokeStyle='rgba(99,102,241,'+(.12*(1-dist/120))+')';
                    ctx.lineWidth=.5; ctx.stroke();
                }
            }
            requestAnimationFrame(draw);
        }
        draw();
    })();
    </script>
    """, unsafe_allow_html=True)

    # Back to landing
    back_col, _ = st.columns([1, 5])
    with back_col:
        if st.button("← Back", key="back_to_landing"):
            st.session_state.show_auth = False
            st.rerun()

    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        st.markdown("""
        <div class="login-anim" style="text-align:center; margin-bottom:24px;">
            <span class="login-icon">🎓</span>
            <div class="login-brand">EduPredict</div>
            <div class="login-tagline">Predict &nbsp;·&nbsp; Improve &nbsp;·&nbsp; Succeed</div>
            <div class="login-divider"></div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑  Login", "✨  Sign Up"])

        # ── LOGIN ──
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="color:#9ca3af;font-size:.82rem;margin-bottom:8px;letter-spacing:.06em;">SELECT YOUR ROLE</p>', unsafe_allow_html=True)
            rc1, rc2 = st.columns(2)
            with rc1:
                is_active = st.session_state.login_role == "Student"
                key = "l_student_active" if is_active else "l_student_btn"
                if st.button("👨‍🎓\n\nStudent", key=key, use_container_width=True):
                    st.session_state.login_role = "Student"; st.rerun()
            with rc2:
                is_active = st.session_state.login_role == "Teacher"
                key = "l_teacher_active" if is_active else "l_teacher_btn"
                if st.button("👩‍🏫\n\nTeacher", key=key, use_container_width=True):
                    st.session_state.login_role = "Teacher"; st.rerun()

            st.markdown(f'<p style="color:#6366f1;font-size:.8rem;margin:10px 0 14px;font-weight:600;">✓ Logging in as: {st.session_state.login_role}</p>', unsafe_allow_html=True)
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

            st.markdown('<p style="text-align:center;color:#374151;font-size:.76rem;margin-top:14px;">Don\'t have an account? Switch to Sign Up ↑</p>', unsafe_allow_html=True)

        # ── SIGNUP ──
        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="color:#9ca3af;font-size:.82rem;margin-bottom:8px;letter-spacing:.06em;">SELECT YOUR ROLE</p>', unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                is_active = st.session_state.signup_role == "Student"
                key = "s_student_active" if is_active else "s_student_btn"
                if st.button("👨‍🎓\n\nStudent", key=key, use_container_width=True):
                    st.session_state.signup_role = "Student"; st.rerun()
            with sc2:
                is_active = st.session_state.signup_role == "Teacher"
                key = "s_teacher_active" if is_active else "s_teacher_btn"
                if st.button("👩‍🏫\n\nTeacher", key=key, use_container_width=True):
                    st.session_state.signup_role = "Teacher"; st.rerun()

            st.markdown(f'<p style="color:#6366f1;font-size:.8rem;margin:10px 0 14px;font-weight:600;">✓ Signing up as: {st.session_state.signup_role}</p>', unsafe_allow_html=True)
            email_s = st.text_input("📧 Gmail address", placeholder="you@gmail.com", key="email_s")
            pass_s  = st.text_input("🔒 Password", type="password", key="pass_s")

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
                        <div class="pw-strength-bar" style="width:{width};background:{color};"></div>
                    </div>
                    <span class="pw-strength-label" style="color:{color};">{label}</span>
                    <div class="pw-checklist">
                        {check_html(has_len,"8+ chars")}{check_html(has_upper,"Uppercase")}
                        {check_html(has_num,"Number")}{check_html(has_spec,"Special (!@#...)")}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#374151;font-size:.76rem;margin-top:4px;">Min 8 chars · uppercase · number · special (!@#$%^&*)</p>', unsafe_allow_html=True)

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

            st.markdown('<p style="text-align:center;color:#374151;font-size:.76rem;margin-top:14px;">Already have an account? Switch to Login ↑</p>', unsafe_allow_html=True)

        users_count = len(load_users())
        st.markdown(f"""
        <div class="login-stats">
            <div class="login-stat-item"><div class="login-stat-val">{users_count}</div><div class="login-stat-lbl">Users</div></div>
            <div class="login-stat-item"><div class="login-stat-val">3</div><div class="login-stat-lbl">ML Models</div></div>
            <div class="login-stat-item"><div class="login-stat-val">6</div><div class="login-stat-lbl">Features</div></div>
        </div>
        <div class="login-footer">EduPredict · Student Performance Analytics</div>
        """, unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════
#  MAIN APP — LOGGED IN
# ══════════════════════════════════════════════
users      = load_users()
history    = load_history()
user_email = st.session_state.user_email
role       = users[user_email]["role"]


# ─────────────────────────────────────────────
#  SIDEBAR — shared helper
# ─────────────────────────────────────────────
def render_sidebar(sections, current_section_key):
    """Renders sidebar brand, nav items, and logout. Returns nothing — sets session state directly."""
    with st.sidebar:
        # Brand
        role_color = "#10b981" if role == "Teacher" else "#f59e0b"
        role_icon  = "👩‍🏫" if role == "Teacher" else "👨‍🎓"
        st.markdown(f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">🎓 EduPredict</div>
            <div class="sidebar-user-email">{user_email}</div>
            <span class="sidebar-role-badge" style="background:{'rgba(16,185,129,.15)' if role=='Teacher' else 'rgba(245,158,11,.15)'};
                  color:{role_color};border:1px solid {'rgba(16,185,129,.3)' if role=='Teacher' else 'rgba(245,158,11,.3)'};">
                {role_icon} {role}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Nav sections
        for group_label, items in sections.items():
            st.markdown(f'<div class="nav-section-label">{group_label}</div>', unsafe_allow_html=True)
            for (icon, label) in items:
                is_active = st.session_state[current_section_key] == label
                btn_key   = f"nav_active_{label}" if is_active else f"nav_{label}"
                prefix    = "▸ " if is_active else "   "
                if st.button(f"{icon}  {prefix}{label}", key=btn_key, use_container_width=True):
                    st.session_state[current_section_key] = label
                    st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Logout
        if st.button("⇠  Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.show_auth = False
            st.rerun()


# ══════════════════════════════════════════════
#  TEACHER DASHBOARD
# ══════════════════════════════════════════════
if role == "Teacher":

    teacher_sections = {
        "MAIN": [
            ("⚡", "Predict"),
            ("📈", "Analysis"),
        ],
        "STUDENTS": [
            ("🏆", "Marks"),
            ("📋", "Progress"),
        ],
        "MODELS": [
            ("🤖", "Model Info"),
        ],
    }
    render_sidebar(teacher_sections, "teacher_section")
    section = st.session_state.teacher_section

    # ── Page header ──
    section_meta = {
        "Predict":    ("⚡ Predict Score",         "Enter student habits to generate a performance prediction"),
        "Analysis":   ("📈 Class Analysis",         "Visual breakdown of class performance and risk levels"),
        "Marks":      ("🏆 Student Marks",          "View predicted scores across all students"),
        "Progress":   ("📋 Student Progress",       "Track improvement trends over multiple attempts"),
        "Model Info": ("🤖 Model Information",      "Compare ML model performance and accuracy metrics"),
    }
    title, subtitle = section_meta.get(section, ("Dashboard", ""))
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)

    # ══ PREDICT SECTION ══
    if section == "Predict":
        st.markdown('<div class="section-header">🔍 Student Input</div>', unsafe_allow_html=True)

        student_email = st.text_input("Student Gmail (to save record)", placeholder="student@gmail.com")
        if student_email and not validate_email(student_email):
            st.error("Enter a valid student Gmail address.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            study_hours     = st.slider("📚 Study Hours / Day",  0, 8,  5)
            social_media    = st.slider("📱 Social Media Hours", 0, 10, 2)
            part_time_job   = st.selectbox("💼 Part-Time Job",   ["No", "Yes"])
        with col2:
            attendance      = st.slider("🏫 Attendance (%)",     0, 100, 60)
            sleep_hours     = st.slider("😴 Sleep Hours",        4, 10,  6)
            extracurricular = st.selectbox("⚽ Extracurricular", ["No", "Yes"])

        predict_clicked = st.button("⚡ Predict Score")

        if predict_clicked:
            ptj   = 1 if part_time_job   == "Yes" else 0
            extra = 1 if extracurricular == "Yes" else 0
            input_dict = {
                "study_hours_per_day":           study_hours,
                "social_media_hours":            social_media,
                "part_time_job":                 ptj,
                "attendance_percentage":         attendance,
                "sleep_hours":                   sleep_hours,
                "extracurricular_participation": extra,
            }
            input_df   = pd.DataFrame([[input_dict[f] for f in features]], columns=features)
            input_data = scaler.transform(input_df)

            all_predictions = {}
            for name, m in models.items():
                pred = m.predict(input_data)[0]
                all_predictions[name] = max(0, min(100, pred))
            prediction = all_predictions[best_model_name]

            risk = "High Risk" if prediction < 50 else ("Medium Risk" if prediction < 70 else "Safe")
            risk_cls = {"Safe": "risk-safe", "Medium Risk": "risk-medium", "High Risk": "risk-high"}[risk]
            colour   = "green" if risk == "Safe" else ("amber" if risk == "Medium Risk" else "red")

            st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f'<div class="metric-tile {colour}"><div class="val">{prediction:.1f}</div><div class="lbl">Predicted Score</div></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="metric-tile"><div class="val" style="font-size:1.3rem"><span class="risk-badge {risk_cls}">{risk}</span></div><div class="lbl" style="margin-top:10px">Risk Level</div></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="metric-tile"><div class="val" style="font-size:1.1rem">{best_model_name}</div><div class="lbl">Best Model Used</div></div>', unsafe_allow_html=True)

            # All model cards
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

            # Bar chart
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(all_predictions.keys(), all_predictions.values(),
                   color=["#6366f1" if n == best_model_name else "#374151" for n in all_predictions],
                   edgecolor="none", width=0.5)
            ax.set_ylabel("Predicted Score")
            ax.set_title("Model Prediction Comparison", color="#e8eaf0", fontweight="bold")
            ax.set_ylim(0, 100)
            ax.axhline(prediction, color="#10b981", linestyle="--", linewidth=1, alpha=0.6)
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

            # Suggestions
            st.markdown('<div class="section-header">💡 Suggestions</div>', unsafe_allow_html=True)
            suggestions = []
            if prediction < 60:       suggestions.append("Focus on studies more consistently")
            if study_hours < 4:        suggestions.append("Increase study hours to at least 4 hours / day")
            if social_media > 5:       suggestions.append("Reduce social media usage")
            if attendance < 50:        suggestions.append("Improve class attendance urgently")
            if sleep_hours < 6:        suggestions.append("Sleep at least 6–7 hours daily")
            if not suggestions:
                suggestions.append("Great job! Keep maintaining your routine! 🎉" if risk == "Safe" else "Try improving consistency in daily habits")
            pills = "".join(f'<span class="suggestion-pill">✔ {s}</span>' for s in suggestions)
            st.markdown(f'<div style="margin-top:8px">{pills}</div>', unsafe_allow_html=True)

            # Save
            if student_email:
                if student_email not in history:
                    history[student_email] = []
                history[student_email].append({
                    "score": float(prediction), "risk": risk,
                    "study_hours": study_hours, "social_media": social_media,
                    "attendance": attendance, "sleep_hours": sleep_hours,
                    "suggestions": suggestions,
                })
                save_history(history)
                st.success(f"✅ Record saved for **{student_email}**")

    # ══ ANALYSIS SECTION ══
    elif section == "Analysis":
        if not history:
            st.info("No student data yet. Make predictions first from the Predict section.")
        else:
            # Build student summary
            student_summary = {}
            for email, records in history.items():
                if not records: continue
                scores_list = [r["score"] for r in records]
                avg_score   = np.mean(scores_list)
                overall_risk = "High Risk" if avg_score < 50 else ("Medium Risk" if avg_score < 70 else "Safe")
                risk_counts  = {"Safe": 0, "Medium Risk": 0, "High Risk": 0}
                for r in records:
                    risk_counts[r["risk"]] = risk_counts.get(r["risk"], 0) + 1
                latest = records[-1]
                student_summary[email] = {
                    "avg_score": round(avg_score, 1), "latest_score": round(scores_list[-1], 1),
                    "attempts": len(records), "overall_risk": overall_risk,
                    "safe_count": risk_counts["Safe"], "medium_count": risk_counts["Medium Risk"],
                    "high_count": risk_counts["High Risk"],
                    "study_hours": latest.get("study_hours","—"), "attendance": latest.get("attendance","—"),
                    "sleep_hours": latest.get("sleep_hours","—"),
                }

            safe_s   = {e:d for e,d in student_summary.items() if d["overall_risk"]=="Safe"}
            medium_s = {e:d for e,d in student_summary.items() if d["overall_risk"]=="Medium Risk"}
            high_s   = {e:d for e,d in student_summary.items() if d["overall_risk"]=="High Risk"}
            all_avgs = [d["avg_score"] for d in student_summary.values()]

            # Summary tiles
            st.markdown('<div class="section-header">📊 Class Summary</div>', unsafe_allow_html=True)
            t1, t2, t3, t4 = st.columns(4)
            for col, val, lbl, cls in [
                (t1, f"{np.mean(all_avgs):.1f}", "Class Avg Score", ""),
                (t2, str(len(safe_s)),   "Safe Students",         "green"),
                (t3, str(len(medium_s)), "Medium Risk Students",  "amber"),
                (t4, str(len(high_s)),   "High Risk Students",    "red"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-tile {cls}"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

            # Charts
            st.markdown('<div class="section-header">📉 Visual Analysis</div>', unsafe_allow_html=True)
            x_vals, y_vals, colors_scatter, trend_scores = [], [], [], []
            for email, records in history.items():
                for r in records:
                    trend_scores.append(r["score"])
                    x_vals.append(r.get("study_hours", 0))
                    y_vals.append(r["score"])
                    overall = student_summary[email]["overall_risk"]
                    colors_scatter.append("#ef4444" if overall=="High Risk" else ("#f59e0b" if overall=="Medium Risk" else "#10b981"))

            ch1, ch2 = st.columns(2)
            with ch1:
                fig1, ax1 = plt.subplots(figsize=(5, 3))
                ax1.plot(trend_scores, color="#6366f1", linewidth=2, marker="o", markersize=4, markerfacecolor="#a5b4fc")
                ax1.fill_between(range(len(trend_scores)), trend_scores, alpha=0.12, color="#6366f1")
                ax1.axhline(70, color="#10b981", linestyle="--", linewidth=1, alpha=0.5, label="Target (70)")
                ax1.set_title("Class Performance Trend", color="#e8eaf0", fontweight="bold")
                ax1.set_xlabel("Predictions (chronological)"); ax1.set_ylabel("Score")
                ax1.legend(facecolor="#13151d", edgecolor="#2d2f3a", labelcolor="#9ca3af")
                fig1.tight_layout(); st.pyplot(fig1); plt.close(fig1)

            with ch2:
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.scatter(x_vals, y_vals, c=colors_scatter, s=80, alpha=0.75, edgecolors="none")
                ax2.set_title("Risk Analysis: Study Hours vs Score", color="#e8eaf0", fontweight="bold")
                ax2.set_xlabel("Study Hours / Day"); ax2.set_ylabel("Predicted Score")
                patches = [
                    mpatches.Patch(color="#10b981", label="Safe (overall)"),
                    mpatches.Patch(color="#f59e0b", label="Medium Risk (overall)"),
                    mpatches.Patch(color="#ef4444", label="High Risk (overall)"),
                ]
                ax2.legend(handles=patches, facecolor="#13151d", edgecolor="#2d2f3a", labelcolor="#9ca3af")
                fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

            # Risk distribution pie
            st.markdown('<div class="section-header">🎯 Risk Distribution</div>', unsafe_allow_html=True)
            _, pie_col, _ = st.columns([1, 2, 1])
            with pie_col:
                risk_counts_total = [len(safe_s), len(medium_s), len(high_s)]
                if sum(risk_counts_total) > 0:
                    fig3, ax3 = plt.subplots(figsize=(5, 4))
                    wedge_colors = ["#10b981", "#f59e0b", "#ef4444"]
                    wedges, texts, autotexts = ax3.pie(
                        risk_counts_total,
                        labels=["Safe", "Medium Risk", "High Risk"],
                        autopct="%1.0f%%",
                        colors=wedge_colors,
                        startangle=140,
                        wedgeprops=dict(edgecolor="#0d0f14", linewidth=2),
                    )
                    for t in texts: t.set_color("#9ca3af")
                    for at in autotexts: at.set_color("#fff"); at.set_fontweight("bold")
                    ax3.set_title("Student Risk Distribution", color="#e8eaf0", fontweight="bold")
                    fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    # ══ MARKS SECTION ══
    elif section == "Marks":
        if not history:
            st.info("No student records yet.")
        else:
            student_summary = {}
            for email, records in history.items():
                if not records: continue
                scores_list  = [r["score"] for r in records]
                avg_score    = np.mean(scores_list)
                overall_risk = "High Risk" if avg_score < 50 else ("Medium Risk" if avg_score < 70 else "Safe")
                risk_counts  = {"Safe": 0, "Medium Risk": 0, "High Risk": 0}
                for r in records:
                    risk_counts[r["risk"]] = risk_counts.get(r["risk"], 0) + 1
                latest = records[-1]
                student_summary[email] = {
                    "avg_score": round(avg_score, 1), "latest_score": round(scores_list[-1], 1),
                    "attempts": len(records), "overall_risk": overall_risk,
                    "safe_count": risk_counts["Safe"], "medium_count": risk_counts["Medium Risk"],
                    "high_count": risk_counts["High Risk"],
                    "study_hours": latest.get("study_hours","—"), "attendance": latest.get("attendance","—"),
                    "sleep_hours": latest.get("sleep_hours","—"),
                }

            safe_s   = {e:d for e,d in student_summary.items() if d["overall_risk"]=="Safe"}
            medium_s = {e:d for e,d in student_summary.items() if d["overall_risk"]=="Medium Risk"}
            high_s   = {e:d for e,d in student_summary.items() if d["overall_risk"]=="High Risk"}

            def student_detail_table(students_dict, risk_label):
                if not students_dict:
                    st.caption(f"No {risk_label} students.")
                    return
                rows = []
                for email, d in students_dict.items():
                    rows.append({
                        "Student Email":     email,
                        "Avg Score":         d["avg_score"],
                        "Latest Score":      d["latest_score"],
                        "Attempts":          d["attempts"],
                        "✅ Safe Pred.":     d["safe_count"],
                        "🟡 Medium Pred.":   d["medium_count"],
                        "🔴 High Pred.":     d["high_count"],
                        "Study Hrs (last)":  d["study_hours"],
                        "Attendance (last)": d["attendance"],
                        "Sleep Hrs (last)":  d["sleep_hours"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-header">📊 Marks by Risk Category</div>', unsafe_allow_html=True)
            exp1, exp2, exp3 = st.columns(3)
            with exp1:
                with st.expander(f"✅ Safe Students ({len(safe_s)})", expanded=True):
                    student_detail_table(safe_s, "Safe")
            with exp2:
                with st.expander(f"🟡 Medium Risk ({len(medium_s)})", expanded=True):
                    student_detail_table(medium_s, "Medium Risk")
            with exp3:
                with st.expander(f"🔴 High Risk ({len(high_s)})", expanded=True):
                    student_detail_table(high_s, "High Risk")

            # Bar chart — avg scores per student
            st.markdown('<div class="section-header">📊 Average Scores Per Student</div>', unsafe_allow_html=True)
            emails_list = list(student_summary.keys())
            avgs_list   = [student_summary[e]["avg_score"] for e in emails_list]
            bar_colors  = ["#10b981" if student_summary[e]["overall_risk"]=="Safe"
                           else ("#f59e0b" if student_summary[e]["overall_risk"]=="Medium Risk" else "#ef4444")
                           for e in emails_list]
            short_labels = [e.split("@")[0] for e in emails_list]

            fig4, ax4 = plt.subplots(figsize=(max(6, len(emails_list)*1.2), 4))
            ax4.bar(short_labels, avgs_list, color=bar_colors, edgecolor="none", width=0.6)
            ax4.axhline(70, color="#6366f1", linestyle="--", linewidth=1.2, alpha=0.6, label="Target (70)")
            ax4.set_ylabel("Avg Predicted Score"); ax4.set_ylim(0, 105)
            ax4.set_title("Student Average Scores", color="#e8eaf0", fontweight="bold")
            plt.xticks(rotation=30, ha="right")
            ax4.legend(facecolor="#13151d", edgecolor="#2d2f3a", labelcolor="#9ca3af")
            fig4.tight_layout(); st.pyplot(fig4); plt.close(fig4)

    # ══ PROGRESS SECTION ══
    elif section == "Progress":
        if not history:
            st.info("No student records yet.")
        else:
            st.markdown('<div class="section-header">📋 Per-Student Progress</div>', unsafe_allow_html=True)
            for email, records in history.items():
                if not records: continue
                scores_list = [r["score"] for r in records]
                avg_score   = np.mean(scores_list)
                latest_risk = records[-1]["risk"]
                risk_cls    = {"Safe": "risk-safe", "Medium Risk": "risk-medium", "High Risk": "risk-high"}[latest_risk]
                trend       = "📈" if len(scores_list) > 1 and scores_list[-1] > scores_list[-2] else ("📉" if len(scores_list) > 1 else "—")

                with st.expander(f"{trend}  {email}   |   Avg: {avg_score:.1f}   |   Latest: {scores_list[-1]:.1f}", expanded=False):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        fig5, ax5 = plt.subplots(figsize=(5, 2.5))
                        ax5.plot(scores_list, color="#6366f1", linewidth=2.2, marker="o", markersize=5, markerfacecolor="#a5b4fc")
                        ax5.fill_between(range(len(scores_list)), scores_list, alpha=0.1, color="#6366f1")
                        ax5.axhline(70, color="#10b981", linestyle="--", linewidth=1, alpha=0.5)
                        ax5.set_ylim(0, 105); ax5.set_ylabel("Score"); ax5.set_xlabel("Attempt #")
                        ax5.set_title(f"Progress: {email.split('@')[0]}", color="#e8eaf0", fontweight="bold")
                        fig5.tight_layout(); st.pyplot(fig5); plt.close(fig5)
                    with col_b:
                        st.markdown(f"""
                        <div style="padding:12px">
                            <div class="metric-tile" style="margin-bottom:10px">
                                <div class="val" style="font-size:1.4rem">{len(scores_list)}</div>
                                <div class="lbl">Attempts</div>
                            </div>
                            <div class="metric-tile">
                                <div style="margin:4px 0"><span class="risk-badge {risk_cls}">{latest_risk}</span></div>
                                <div class="lbl" style="margin-top:8px">Latest Risk</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    if records[-1].get("suggestions"):
                        pills = "".join(f'<span class="suggestion-pill" style="font-size:.78rem">✔ {s}</span>' for s in records[-1]["suggestions"])
                        st.markdown(f'<div style="margin-top:8px">{pills}</div>', unsafe_allow_html=True)

    # ══ MODEL INFO SECTION ══
    elif section == "Model Info":
        st.markdown('<div class="section-header">📋 Model Comparison Table</div>', unsafe_allow_html=True)
        st.dataframe(
            loaded["results"].style
                .highlight_min(subset=["rmse"], color="#1a2a1a")
                .highlight_max(subset=["r2"],   color="#1a2a1a")
                .format({"rmse": "{:.4f}", "r2": "{:.4f}"}),
            use_container_width=True,
        )
        st.info(f"**{best_model_name}** selected as best model — lowest RMSE = highest prediction accuracy.")

        st.markdown('<div class="section-header">📊 Model Accuracy Comparison</div>', unsafe_allow_html=True)
        results_df = loaded["results"]
        ch1, ch2 = st.columns(2)
        with ch1:
            fig6, ax6 = plt.subplots(figsize=(5, 3))
            ax6.bar(results_df["model"], results_df["rmse"],
                    color=["#6366f1" if n==best_model_name else "#374151" for n in results_df["model"]],
                    edgecolor="none", width=0.5)
            ax6.set_ylabel("RMSE (lower = better)"); ax6.set_title("RMSE by Model", color="#e8eaf0", fontweight="bold")
            fig6.tight_layout(); st.pyplot(fig6); plt.close(fig6)
        with ch2:
            fig7, ax7 = plt.subplots(figsize=(5, 3))
            ax7.bar(results_df["model"], results_df["r2"],
                    color=["#10b981" if n==best_model_name else "#374151" for n in results_df["model"]],
                    edgecolor="none", width=0.5)
            ax7.set_ylabel("R² Score (higher = better)"); ax7.set_title("R² by Model", color="#e8eaf0", fontweight="bold")
            fig7.tight_layout(); st.pyplot(fig7); plt.close(fig7)

        # Feature info
        st.markdown('<div class="section-header">🔑 Features Used</div>', unsafe_allow_html=True)
        feat_col1, feat_col2 = st.columns(2)
        feat_icons = {"study_hours_per_day":"📚","social_media_hours":"📱","part_time_job":"💼",
                      "attendance_percentage":"🏫","sleep_hours":"😴","extracurricular_participation":"⚽"}
        for i, feat in enumerate(features):
            col = feat_col1 if i % 2 == 0 else feat_col2
            with col:
                st.markdown(
                    f'<div class="metric-tile" style="margin-bottom:10px;text-align:left;padding:14px 18px">'
                    f'<span style="font-size:1.2rem">{feat_icons.get(feat,"📌")}</span> '
                    f'<span style="font-family:Syne,sans-serif;font-weight:700;color:#e8eaf0;">{feat.replace("_"," ").title()}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════
#  STUDENT DASHBOARD
# ══════════════════════════════════════════════
else:
    student_sections = {
        "MY PERFORMANCE": [
            ("📊", "Overview"),
            ("📈", "Progress"),
        ],
        "INSIGHTS": [
            ("💡", "Suggestions"),
            ("📋", "History"),
        ],
    }
    render_sidebar(student_sections, "student_section")
    section   = st.session_state.student_section
    user_data = history.get(user_email, [])

    section_meta = {
        "Overview":    ("📊 My Overview",      "Your latest predicted score and habit snapshot"),
        "Progress":    ("📈 My Progress",       "Performance trend across all your attempts"),
        "Suggestions": ("💡 Suggestions",       "Personalised tips to improve your score"),
        "History":     ("📋 Attempt History",   "Full record of every prediction made for you"),
    }
    title, subtitle = section_meta.get(section, ("Dashboard", ""))
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)

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
        latest     = user_data[-1]
        score      = latest["score"]
        risk       = latest["risk"]
        risk_cls   = {"Safe": "risk-safe", "Medium Risk": "risk-medium", "High Risk": "risk-high"}[risk]
        score_color = "green" if risk == "Safe" else ("amber" if risk == "Medium Risk" else "red")
        scores_all  = [x["score"] for x in user_data]
        avg_score   = np.mean(scores_all)

        # ══ OVERVIEW ══
        if section == "Overview":
            st.markdown('<div class="section-header">📊 Latest Result</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-tile {score_color}"><div class="val">{score:.1f}</div><div class="lbl">Latest Score</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-tile"><div class="val">{avg_score:.1f}</div><div class="lbl">Average Score</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-tile"><div class="val">{len(user_data)}</div><div class="lbl">Total Attempts</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-tile"><div style="margin-top:6px"><span class="risk-badge {risk_cls}">{risk}</span></div><div class="lbl" style="margin-top:10px">Risk Level</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">📋 Latest Habit Snapshot</div>', unsafe_allow_html=True)
            h1, h2, h3, h4 = st.columns(4)
            for col, val, lbl in [
                (h1, f'{latest.get("study_hours","—")}h',  "Study Hours/Day"),
                (h2, f'{latest.get("social_media","—")}h', "Social Media"),
                (h3, f'{latest.get("attendance","—")}%',   "Attendance"),
                (h4, f'{latest.get("sleep_hours","—")}h',  "Sleep Hours"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-tile"><div class="val" style="font-size:1.5rem">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

            # Mini progress chart
            st.markdown('<div class="section-header">📈 Quick Progress View</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(scores_all, color="#6366f1", linewidth=2.5, marker="o", markersize=6, markerfacecolor="#a5b4fc", zorder=3)
            ax.fill_between(range(len(scores_all)), scores_all, alpha=0.1, color="#6366f1")
            ax.axhline(70, color="#10b981", linestyle="--", linewidth=1, alpha=0.5, label="Target (70)")
            ax.set_title("Performance Over Attempts", color="#e8eaf0", fontweight="bold")
            ax.set_xlabel("Attempt #"); ax.set_ylabel("Predicted Score"); ax.set_ylim(0, 105)
            ax.legend(facecolor="#13151d", edgecolor="#2d2f3a", labelcolor="#9ca3af")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ══ PROGRESS ══
        elif section == "Progress":
            st.markdown('<div class="section-header">📈 Performance Over Time</div>', unsafe_allow_html=True)

            # Score trend
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(scores_all, color="#6366f1", linewidth=2.5, marker="o", markersize=7, markerfacecolor="#a5b4fc", zorder=3)
            ax.fill_between(range(len(scores_all)), scores_all, alpha=0.1, color="#6366f1")
            ax.axhline(70, color="#10b981", linestyle="--", linewidth=1.2, alpha=0.5, label="Target (70)")
            ax.axhline(avg_score, color="#f59e0b", linestyle=":", linewidth=1.2, alpha=0.6, label=f"Your Avg ({avg_score:.1f})")
            ax.set_title("Your Score Progress", color="#e8eaf0", fontweight="bold")
            ax.set_xlabel("Attempt #"); ax.set_ylabel("Predicted Score"); ax.set_ylim(0, 105)
            ax.legend(facecolor="#13151d", edgecolor="#2d2f3a", labelcolor="#9ca3af")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            # Stats
            st.markdown('<div class="section-header">🏅 Performance Stats</div>', unsafe_allow_html=True)
            best_score = max(scores_all)
            worst_score = min(scores_all)
            improvement = scores_all[-1] - scores_all[0] if len(scores_all) > 1 else 0
            s1, s2, s3, s4 = st.columns(4)
            for col, val, lbl, cls in [
                (s1, f"{best_score:.1f}",  "Best Score",    "green"),
                (s2, f"{worst_score:.1f}", "Lowest Score",  "red"),
                (s3, f"{avg_score:.1f}",   "Your Average",  ""),
                (s4, f"{improvement:+.1f}","Overall Change","green" if improvement >= 0 else "red"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-tile {cls}"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

            # Risk history
            st.markdown('<div class="section-header">🎯 Risk Level Over Attempts</div>', unsafe_allow_html=True)
            risk_map = {"Safe": 3, "Medium Risk": 2, "High Risk": 1}
            risk_vals  = [risk_map[r["risk"]] for r in user_data]
            risk_labels = [r["risk"] for r in user_data]
            fig2, ax2 = plt.subplots(figsize=(9, 2.5))
            colors_risk = ["#10b981" if v==3 else ("#f59e0b" if v==2 else "#ef4444") for v in risk_vals]
            ax2.bar(range(len(risk_vals)), risk_vals, color=colors_risk, edgecolor="none", width=0.6)
            ax2.set_yticks([1,2,3]); ax2.set_yticklabels(["High Risk","Medium Risk","Safe"])
            ax2.set_xlabel("Attempt #"); ax2.set_title("Risk Level Per Attempt", color="#e8eaf0", fontweight="bold")
            fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

        # ══ SUGGESTIONS ══
        elif section == "Suggestions":
            suggestions = latest.get("suggestions", [])
            if not suggestions:
                st.markdown('<span class="suggestion-pill" style="font-size:1rem;padding:10px 20px">✔ Keep up the great work! 🎉</span>', unsafe_allow_html=True)
            else:
                for s in suggestions:
                    st.markdown(
                        f'<div class="card card-accent" style="padding:18px 24px;margin-bottom:12px;">'
                        f'<span style="font-size:1rem;color:#a5b4fc;font-weight:600">✔ {s}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Habit vs target comparison
            st.markdown('<div class="section-header">🎯 Habit vs Target</div>', unsafe_allow_html=True)
            targets = {
                "Study Hours": (latest.get("study_hours", 0), 6),
                "Attendance":  (latest.get("attendance", 0) / 100 * 8, 8),
                "Sleep Hours": (latest.get("sleep_hours", 0), 7),
                "Social Media": (latest.get("social_media", 0), 3),
            }
            fig3, ax3 = plt.subplots(figsize=(7, 3))
            x_pos      = range(len(targets))
            actuals    = [v[0] for v in targets.values()]
            target_vals = [v[1] for v in targets.values()]
            x_list = list(x_pos)
            bar_w  = 0.35
            ax3.bar([x - bar_w/2 for x in x_list], actuals, width=bar_w, color="#6366f1", label="Current", edgecolor="none")
            ax3.bar([x + bar_w/2 for x in x_list], target_vals, width=bar_w, color="#374151", label="Target",  edgecolor="none")
            ax3.set_xticks(x_list); ax3.set_xticklabels(targets.keys())
            ax3.set_title("Your Habits vs Recommended Targets", color="#e8eaf0", fontweight="bold")
            ax3.legend(facecolor="#13151d", edgecolor="#2d2f3a", labelcolor="#9ca3af")
            fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)

        # ══ HISTORY ══
        elif section == "History":
            st.markdown('<div class="section-header">📋 All Prediction Records</div>', unsafe_allow_html=True)
            rows = []
            for i, r in enumerate(user_data, 1):
                rows.append({
                    "Attempt":      i,
                    "Score":        round(r["score"], 1),
                    "Risk":         r["risk"],
                    "Study Hrs":    r.get("study_hours", "—"),
                    "Social Media": r.get("social_media", "—"),
                    "Attendance":   r.get("attendance",  "—"),
                    "Sleep Hrs":    r.get("sleep_hours", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Habit trend charts
            if len(user_data) > 1:
                st.markdown('<div class="section-header">📊 Habit Trends Over Time</div>', unsafe_allow_html=True)
                fig4, axes = plt.subplots(1, 3, figsize=(12, 3))
                for ax, key, label, color in [
                    (axes[0], "study_hours",  "Study Hours",  "#6366f1"),
                    (axes[1], "attendance",   "Attendance %", "#10b981"),
                    (axes[2], "sleep_hours",  "Sleep Hours",  "#f59e0b"),
                ]:
                    vals = [r.get(key, 0) for r in user_data]
                    ax.plot(vals, color=color, linewidth=2, marker="o", markersize=5)
                    ax.fill_between(range(len(vals)), vals, alpha=0.1, color=color)
                    ax.set_title(label, color="#e8eaf0", fontweight="bold")
                    ax.set_xlabel("Attempt #")
                fig4.tight_layout(); st.pyplot(fig4); plt.close(fig4)