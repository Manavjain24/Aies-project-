"""Minimal Streamlit frontend. [NOT EXECUTED]"""
import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title='Evolving Heuristics',layout='wide')
st.title('LLM-driven heuristic evolution')
st.caption('The current reproducible runner is an offline 8-Puzzle surrogate; LLM calls are not connected.')
results=Path(__file__).parent/'results'/'results.csv'
if results.exists():
    df=pd.read_csv(results); st.dataframe(df,use_container_width=True)
    st.bar_chart(df.groupby('setting')['test_mean_expansions'].mean())
else:
    st.info('Run experiment.py first.')
