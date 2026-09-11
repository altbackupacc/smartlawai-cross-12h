"""Minimal Streamlit skeleton for M0: upload -> ingest, then doc_id + question ->
ask. A skeleton, not a real UI -- run: streamlit run ui/app.py"""
from __future__ import annotations

import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smartlawai.adapters.factory import get_backend
from smartlawai.pipeline import Pipeline, build_production_pipeline
from smartlawai.scope import Scope

st.set_page_config(page_title="SmartLawAI (M0 skeleton)")
st.title("SmartLawAI — M0 skeleton")

owner_id = st.text_input("Owner ID", value="anon")


@st.cache_resource
def _pipeline() -> Pipeline:
    return build_production_pipeline(get_backend())


pipeline = _pipeline()

st.header("1. Ingest a document")
uploaded = st.file_uploader("Upload a document", type=["txt", "md", "pdf"])
if uploaded and st.button("Ingest"):
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name
    doc_id, trace = pipeline.ingest(path, doc_type="JUDGMENT", source="ui",
                                    owner_id=owner_id)
    st.success(f"doc_id: {doc_id}")
    st.json(trace.to_dict())

st.header("2. Ask a scoped question")
doc_id_input = st.text_input("doc_id (from step 1)")
question = st.text_input("Question")
if st.button("Ask") and doc_id_input and question:
    scope = Scope(doc_ids=(doc_id_input,), owner_id=owner_id)
    result = pipeline.ask(question, scope)
    st.write(f"**Decision:** {result.decision}")
    st.write(f"**Answer:** {result.answer}")
    st.json(result.trace.to_dict())
