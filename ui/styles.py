"""Look and feel for the Streamlit UI: one CSS block plus small HTML helpers (hero, step tracker).

Colours: deep navy background, violet to cyan accents. All visual choices live here so the screens stay about content.
"""

import streamlit as st

STEPS = ["Topic", "Outline", "Report", "Analysis", "Chat"]
# Which tracker step each screen belongs to (index into STEPS)
PAGE_STEP = {"topic": 0, "outline": 1, "writing": 2, "report": 2, "analysis": 3, "chat": 4}

CSS = """
<style>
:root {
  --violet: #8b5cf6; --cyan: #22d3ee; --pink: #f472b6; --ink: #e8ecff; --muted: #94a0cc;
  --card: rgba(255,255,255,0.045); --line: rgba(255,255,255,0.10);
}

/* animated aurora background */
.stApp {
  background:
    radial-gradient(900px 500px at 8% -5%, rgba(139,92,246,0.28), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(34,211,238,0.18), transparent 60%),
    radial-gradient(700px 600px at 50% 110%, rgba(244,114,182,0.14), transparent 60%),
    #0b1020;
  background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 980px; animation: rise .6s ease both; }

@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes shimmer { to { background-position: 200% center; } }
@keyframes floaty { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(139,92,246,.55); } 70% { box-shadow: 0 0 0 12px rgba(139,92,246,0); } }

/* hero */
.hero { text-align: center; padding: 1.2rem 0 1.6rem; }
.hero .logo { font-size: 3rem; display: inline-block; animation: floaty 3.5s ease-in-out infinite; }
.hero h1 {
  font-size: 3.4rem; font-weight: 800; letter-spacing: -0.03em; margin: .2rem 0 .3rem;
  background: linear-gradient(90deg, var(--violet), var(--cyan), var(--pink), var(--violet));
  background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; animation: shimmer 6s linear infinite;
}
.hero p { color: var(--muted); font-size: 1.1rem; margin: 0; }

.page-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: .2rem 0 .1rem; }
.page-sub { color: var(--muted); margin-bottom: 1.1rem; }
.topic-pill {
  display: inline-block; padding: .15rem .8rem; border-radius: 999px; font-weight: 600;
  background: linear-gradient(90deg, rgba(139,92,246,.25), rgba(34,211,238,.25)); border: 1px solid var(--line);
}

/* step tracker */
.steps { display: flex; justify-content: center; align-items: center; gap: .4rem; margin: 0 0 1.6rem; flex-wrap: wrap; }
.step { display: flex; align-items: center; gap: .5rem; color: var(--muted); font-weight: 600; font-size: .9rem; }
.step .dot {
  width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-size: .8rem;
  background: var(--card); border: 1px solid var(--line);
}
.step.done .dot { background: linear-gradient(135deg, var(--violet), var(--cyan)); color: #fff; border: none; }
.step.now { color: var(--ink); }
.step.now .dot { background: linear-gradient(135deg, var(--violet), var(--pink)); color: #fff; border: none; animation: pulse 2s infinite; }
.step-line { width: 34px; height: 2px; background: var(--line); border-radius: 2px; }
.step-line.done { background: linear-gradient(90deg, var(--violet), var(--cyan)); }

/* cards: bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card); border: 1px solid var(--line) !important; border-radius: 16px;
  backdrop-filter: blur(8px); transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
  animation: rise .5s ease both;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-2px); border-color: rgba(139,92,246,.6) !important;
  box-shadow: 0 10px 30px -12px rgba(139,92,246,.55);
}

/* inputs */
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="base-input"] {
  background: rgba(255,255,255,0.05) !important; border-radius: 12px !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
  box-shadow: 0 0 0 2px rgba(139,92,246,.7), 0 0 24px rgba(139,92,246,.35);
}
.stTextInput label, .stTextArea label { color: var(--muted) !important; font-weight: 600; }

/* buttons */
.stButton > button {
  border-radius: 12px; border: 1px solid var(--line); background: var(--card); color: var(--ink);
  font-weight: 600; transition: all .2s ease;
}
.stButton > button:hover { border-color: var(--violet); transform: translateY(-1px); box-shadow: 0 6px 20px -8px rgba(139,92,246,.7); }
.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, var(--violet), var(--cyan)); background-size: 150% auto; color: #fff; border: none;
  padding: .6rem 1.6rem; font-size: 1.02rem; box-shadow: 0 8px 28px -10px rgba(139,92,246,.9);
}
.stButton > button[kind="primary"]:hover { background-position: right center; transform: translateY(-2px) scale(1.02); }
.stButton > button:disabled { opacity: .4; transform: none; box-shadow: none; }

/* section number badge */
.badge {
  display: inline-grid; place-items: center; width: 30px; height: 30px; border-radius: 9px; margin-right: .5rem;
  background: linear-gradient(135deg, var(--violet), var(--pink)); color: #fff; font-weight: 800; font-size: .9rem;
}
.sec-head { font-weight: 700; color: var(--ink); margin-bottom: .2rem; }

/* working screen checklist */
.work-item {
  display: flex; align-items: center; gap: .7rem; padding: .7rem 1rem; margin: .45rem 0; border-radius: 12px;
  background: var(--card); border: 1px solid var(--line); animation: rise .5s ease both;
}
.work-item .spin {
  width: 16px; height: 16px; border-radius: 50%; border: 2px solid var(--line); border-top-color: var(--cyan);
  animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* sidebar */
section[data-testid="stSidebar"] { background: rgba(11,16,32,.85); border-right: 1px solid var(--line); }
.brand { font-weight: 800; font-size: 1.3rem; background: linear-gradient(90deg, var(--violet), var(--cyan));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }

#MainMenu, footer { visibility: hidden; }
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="logo">🧭</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def step_tracker(page: str) -> None:
    current = PAGE_STEP.get(page, 0)
    parts = []
    for index, name in enumerate(STEPS):
        state = "done" if index < current else "now" if index == current else ""
        mark = "✓" if index < current else str(index + 1)
        parts.append(f'<div class="step {state}"><span class="dot">{mark}</span>{name}</div>')
        if index < len(STEPS) - 1:
            parts.append(f'<div class="step-line {"done" if index < current else ""}"></div>')
    st.markdown(f'<div class="steps">{"".join(parts)}</div>', unsafe_allow_html=True)
