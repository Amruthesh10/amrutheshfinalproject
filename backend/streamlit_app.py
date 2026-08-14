import streamlit as st
from graph.workflow import app

st.set_page_config(
    page_title="Smart E-Commerce Assistant",
    page_icon="🛒"
)

st.title("🛒 Smart E-Commerce Assistant")

query = st.text_input(
    "What are you looking for?"
)

if st.button("Search"):

    if query:

        state = {
            "query": query
        }

        with st.spinner("Analyzing products..."):

            result = app.invoke(state)

        st.markdown("### Recommendation")

        st.write(result["response"])