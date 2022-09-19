import streamlit as st

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "⛽ Predicta.Oil";
                margin-left: 20px;
                margin-top: 10px;
                margin-bottom: 10px;
                font-size: 30px;
                position: relative;
                top: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
