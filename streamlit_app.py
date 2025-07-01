import streamlit as st
from rag_pipeline import app  # your LangGraph graph

# Initialize chat history if not set
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="LangGraph RAG Chat", layout="wide")
st.title("🧠 LangGraph RAG Chatbot")

# Text input
question = st.text_input("Ask a question about your PDF:")

if st.button("Submit") and question:
    # Prepare state for LangGraph
    input_state = {
        "question": question,
        "chat_history": st.session_state.chat_history
    }

    # Run LangGraph app
    result = app.invoke(input_state)

    # Update session history
    st.session_state.chat_history = result["chat_history"]

    # Display answer
    st.markdown("### 📌 Answer")
    st.write(result["answer"])

    # Display history
    with st.expander("💬 Full Chat History"):
        for turn in result["chat_history"]:
            st.markdown(f"**You:** {turn['question']}")
            st.markdown(f"**Bot:** {turn['answer']}")
