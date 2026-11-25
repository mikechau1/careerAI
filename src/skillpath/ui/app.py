"""SkillPath AI Streamlit app. Run with: streamlit run src/skillpath/ui/app.py

Placeholder until M3-05 (dashboard) and M3-06 (chat tab) are built.
"""

import streamlit as st

st.set_page_config(page_title="SkillPath AI", layout="wide")
st.title("SkillPath AI")
st.caption("Agentic hybrid recommender for career and learning paths")

dashboard, chat = st.tabs(["Dashboard", "Career assistant"])
with dashboard:
    st.info("Resume upload, job matches, skill gaps and courses will appear here (M3-05).")
with chat:
    st.info("Conversational assistant coming in M3-06.")
