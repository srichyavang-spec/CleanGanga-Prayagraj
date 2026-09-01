import streamlit as st
from google import genai

st.title("CleanGanga – Gemini Test")

st.write("Streamlit is working ✅")

if st.button("Test Gemini"):
    try:
        client = genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"]
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello in one sentence."
        )

        st.success("Gemini is working!")
        st.write(response.text)

    except Exception as e:
        st.error("Gemini request failed")
        st.exception(e)