from wellscope_robust_app import main
import streamlit as st
from wellscope_core import AnalysisError

try:
    main()
except AnalysisError as exc:
    st.error(str(exc))
