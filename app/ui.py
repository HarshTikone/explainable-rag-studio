"""Shared presentation primitives for the Streamlit application."""
from __future__ import annotations

from html import escape
from typing import Iterable

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#e8edf7; --muted:#8f9bb3; --panel:rgba(19,27,44,.82); --line:rgba(148,163,184,.16); --cyan:#32d6c7; --violet:#8d7bff; --amber:#ffbf69; }
html,body,[class*="css"]{font-family:'DM Sans',sans-serif} h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.03em}
.stApp{background:radial-gradient(circle at 82% -8%,rgba(87,70,200,.18),transparent 34rem),radial-gradient(circle at 15% 12%,rgba(17,179,166,.10),transparent 30rem),#080d18;color:var(--ink)}
[data-testid="stHeader"]{background:rgba(8,13,24,.7);backdrop-filter:blur(14px)} [data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1220,#090f1b);border-right:1px solid var(--line)}
[data-testid="stSidebarNav"] span{font-weight:600} [data-testid="stSidebarNav"] a{border-radius:10px;margin:2px 8px} [data-testid="stSidebarNav"] a:hover{background:rgba(50,214,199,.08)}
.block-container{max-width:1280px;padding-top:2rem;padding-bottom:4rem}.brand-lockup{padding:.5rem .25rem 1.25rem}.brand-mark{display:inline-grid;place-items:center;width:36px;height:36px;margin-right:10px;border-radius:11px;color:#07101a;font:700 17px 'Space Grotesk';background:linear-gradient(135deg,var(--cyan),#7cf0a8);box-shadow:0 0 28px rgba(50,214,199,.22)}
.brand-name{color:#f4f7fb;font:600 16px 'Space Grotesk';vertical-align:middle}.brand-caption{color:var(--muted);font-size:11px;margin:8px 0 0 48px;letter-spacing:.08em;text-transform:uppercase}
.hero{padding:.4rem 0 1.4rem}.eyebrow{color:var(--cyan);font-size:.73rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase}.hero h1{color:#f5f7fb;font-size:clamp(2rem,4vw,3.35rem);line-height:1.04;margin:.45rem 0 .7rem}.hero p{color:var(--muted);font-size:1.05rem;line-height:1.65;max-width:760px;margin:0}.hero-line{height:1px;margin-top:1.45rem;background:linear-gradient(90deg,var(--cyan),var(--violet),transparent 75%);opacity:.65}
.metric-card,.feature-card{height:100%;padding:1.05rem 1.1rem;border:1px solid var(--line);border-radius:15px;background:linear-gradient(145deg,rgba(20,29,48,.94),rgba(12,19,33,.84));box-shadow:0 16px 45px rgba(0,0,0,.14)}
.metric-label{color:var(--muted);font-size:.72rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase}.metric-value{color:#f6f8fc;font:600 1.65rem 'Space Grotesk';margin:.3rem 0 .1rem}.metric-detail{color:var(--muted);font-size:.82rem}.feature-card h3{color:#f0f4fa;font-size:1.03rem;margin:.25rem 0 .5rem}.feature-card p{color:var(--muted);font-size:.88rem;line-height:1.55;margin:0}.feature-index{color:var(--cyan);font:600 .74rem 'Space Grotesk';letter-spacing:.09em}
.status-row{display:flex;flex-wrap:wrap;gap:8px;margin:.2rem 0 1rem}.status-pill{display:inline-flex;align-items:center;gap:7px;color:#c6d0e1;padding:6px 10px;border:1px solid var(--line);border-radius:999px;background:rgba(17,26,45,.72);font-size:.76rem}.status-dot{width:7px;height:7px;border-radius:50%;background:var(--muted)}.status-dot.ok{background:var(--cyan);box-shadow:0 0 9px rgba(50,214,199,.7)}.status-dot.warn{background:var(--amber)}
[data-testid="stMetric"]{padding:1rem;border:1px solid var(--line);border-radius:14px;background:var(--panel)} [data-testid="stFileUploader"]{border:1px dashed rgba(50,214,199,.34);border-radius:14px;padding:.35rem;background:rgba(17,26,45,.5)} [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden} [data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:rgba(17,26,45,.55)}
.stButton>button{min-height:2.75rem;border-radius:10px;border:1px solid rgba(50,214,199,.25);font-weight:700;transition:transform .16s ease,box-shadow .16s ease}.stButton>button[kind="primary"]{color:#061413;background:linear-gradient(135deg,#32d6c7,#72e6b0);border:0}.stButton>button:hover{transform:translateY(-1px);box-shadow:0 9px 24px rgba(50,214,199,.14)}
.section-heading{color:#eef3fb;font:600 1.15rem 'Space Grotesk';margin:1.6rem 0 .25rem}.section-copy{color:var(--muted);margin:0 0 1rem;font-size:.9rem}.footer-note{color:#66738b;font-size:.75rem;padding-top:2.5rem}
@media(max-width:700px){.block-container{padding:1.2rem 1rem 3rem}.hero h1{font-size:2.05rem}.hero p{font-size:.95rem}}
</style>
"""

def configure_page(title: str, icon: str = "◈") -> None:
    st.set_page_config(page_title=f"{title} · Explainable RAG Studio", page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand-lockup"><span class="brand-mark">R</span><span class="brand-name">RAG Studio</span><div class="brand-caption">Reliability workbench</div></div>', unsafe_allow_html=True)

def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<section class="hero"><div class="eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1><p>{escape(description)}</p><div class="hero-line"></div></section>', unsafe_allow_html=True)

def section(title: str, description: str = "") -> None:
    st.markdown(f'<div class="section-heading">{escape(title)}</div><div class="section-copy">{escape(description)}</div>', unsafe_allow_html=True)

def metric_card(label: str, value: str, detail: str) -> None:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{escape(label)}</div><div class="metric-value">{escape(value)}</div><div class="metric-detail">{escape(detail)}</div></div>', unsafe_allow_html=True)

def feature_card(index: str, title: str, description: str) -> None:
    st.markdown(f'<div class="feature-card"><div class="feature-index">{escape(index)}</div><h3>{escape(title)}</h3><p>{escape(description)}</p></div>', unsafe_allow_html=True)

def status_pills(items: Iterable[tuple[str, bool]]) -> None:
    pills = "".join(f'<span class="status-pill"><span class="status-dot {"ok" if active else "warn"}"></span>{escape(label)}</span>' for label, active in items)
    st.markdown(f'<div class="status-row">{pills}</div>', unsafe_allow_html=True)

def footer() -> None:
    st.markdown('<div class="footer-note">Explainable RAG Studio · Transparent retrieval, grounded answers, measurable quality.</div>', unsafe_allow_html=True)
