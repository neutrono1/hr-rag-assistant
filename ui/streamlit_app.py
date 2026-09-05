import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="HR Policy Assistant", page_icon="📋")
st.title("📋 HR Policy Assistant")
st.caption("Answers are grounded only in uploaded policy documents, with citations.")

with st.sidebar:
    st.header("Admin")
    is_admin = st.checkbox("I'm an admin (hardcoded flag, not real auth)")
    if is_admin:
        uploaded = st.file_uploader("Upload a policy (.md, .txt, .pdf)", type=["md", "txt", "pdf"])
        if uploaded and st.button("Upload & index"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            resp = requests.post(f"{API_URL}/admin/documents", files=files, headers={"X-Role": "admin"})
            if resp.ok:
                data = resp.json()
                st.success(f"Indexed {data['filename']} into {data['num_chunks']} chunks.")
            else:
                st.error(resp.json().get("detail", resp.text))

    st.divider()
    st.subheader("Indexed documents")
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=10).json()
        if docs:
            for d in docs:
                st.write(f"• {d['filename']} ({d['num_chunks']} chunks)")
        else:
            st.write("No documents indexed yet.")
    except requests.RequestException:
        st.warning("Cannot reach the API. Is it running?")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

question = st.chat_input("Ask a policy question, e.g. 'How many casual leave days carry forward?'")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            st.write(data["answer"])
            if data["citations"]:
                st.markdown("**Sources:**")
                for c in data["citations"]:
                    st.markdown(f"- `{c['document']}` — {c['section']}")
            elif not data["sufficient"]:
                st.caption("No citations — question could not be answered from the uploaded policies.")
            st.session_state.history.append({"role": "assistant", "content": data["answer"]})
        except requests.RequestException as e:
            st.error(f"Could not reach the API: {e}")
