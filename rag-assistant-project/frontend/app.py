import streamlit as st
from api_client import ask
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Academic Paper Assistant", page_icon="📚")
st.title("📚 Academic Paper Assistant")
st.caption("Ask grounded questions about the indexed research-paper corpus.")

# Setup session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask about a paper, figure, table, or equation..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and drafting a grounded answer..."):
            try:
                # Call the backend via the client
                result = ask(question)

                # Format the sources for display
                source_list = []
                for s in result.get("sources", []):
                    source_info = f"- Doc: {s.get('document_id')}, Chunk: {s.get('chunk_id')}"
                    source_list.append(source_info)

                sources_text = "\n".join(source_list) if source_list else "No specific sources cited."

                full_content = f"{result.get('answer', 'No answer generated.')}\n\n**Retrieved sources**\n{sources_text}"

                st.markdown(full_content)
                st.session_state.messages.append({"role": "assistant", "content": full_content})

            except Exception as exc:
                # If the exception message contains "Could not connect" or "ConnectionError", it's a real server issue.
                # Otherwise, it's likely a model response we should just display.
                error_msg = str(exc)
                if "Could not connect" in error_msg or "ConnectionError" in error_msg or "Server error" in error_msg:
                    st.error(f"I couldn't reach the assistant service. Check that the backend is running. ({exc})")
                else:
                    # Display the actual response from the service (e.g., "I don't know...")
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
