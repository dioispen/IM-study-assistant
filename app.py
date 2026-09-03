"""Streamlit's entry point: `streamlit run app.py`.

Two lines, and deliberately. Streamlit re-executes this script from source on
every interaction, so a name defined here is defined afresh each run -- a
factory hook living in this file could never be the one a test had substituted
before the run, and the injection the app suite depends on would quietly become
impossible. The page is therefore in interface.py, which is imported once and
can be patched, and this hands Streamlit something to run.
"""

from interface import render_page

render_page()
