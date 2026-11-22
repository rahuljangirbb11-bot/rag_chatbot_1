import os
import tempfile
import streamlit as st

from database import create_vectorstore
from chains import create_rag_chain, create_memory


# -------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about its contents.")


# -------------------------------
# 2. SESSION STATE INITIALIZATION
# -------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "memory" not in st.session_state:
    st.session_state.memory = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------
# 3. SIDEBAR - PDF UPLOAD
# -------------------------------

with st.sidebar:

    st.header("Upload Your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    process_button = st.button(
        "Process PDF",
        use_container_width=True
    )

    if process_button:

        if uploaded_file is None:
            st.warning("Please upload a PDF first.")

        else:

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                with st.spinner("Processing your PDF..."):

                    vectorstore, page_count, chunk_count = (
                        create_vectorstore(temp_path)
                    )

                    rag_chain = create_rag_chain(vectorstore)
                    memory = create_memory()

                    st.session_state.vectorstore = vectorstore
                    st.session_state.rag_chain = rag_chain
                    st.session_state.memory = memory
                    st.session_state.messages = []

                st.success("PDF processed successfully!")

                st.write(f"Pages processed: {page_count}")
                st.write(f"Text chunks created: {chunk_count}")

            except Exception as e:

                st.error(f"Error processing PDF: {e}")

            finally:

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# -------------------------------
# 4. CLEAR CONVERSATION
# -------------------------------

if st.session_state.rag_chain is not None:

    if st.sidebar.button("Clear Conversation"):

        st.session_state.memory = create_memory()
        st.session_state.messages = []

        st.rerun()


# -------------------------------
# 5. DISPLAY CHAT HISTORY
# -------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------------
# 6. CHAT INPUT AND RESPONSE
# -------------------------------

user_question = st.chat_input(
    "Ask something about your PDF..."
)

if user_question:

    if st.session_state.rag_chain is None:

        st.warning("Please upload and process a PDF first.")

    else:

        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        try:

            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    chat_history = (
                        st.session_state.memory
                        .load_memory_variables({})["chat_history"]
                    )

                    response = st.session_state.rag_chain.invoke({
                        "input": user_question,
                        "chat_history": chat_history
                    })

                    answer = response["answer"]

                    st.markdown(answer)

                    with st.expander("View retrieved sources"):

                        for doc in response["context"]:

                            page = doc.metadata.get(
                                "page",
                                "Unknown"
                            )

                            st.write(f"Page: {page + 1 if isinstance(page, int) else page}")

                            st.write(
                                doc.page_content[:300] + "..."
                            )

                            st.divider()

            st.session_state.memory.save_context(
                {"input": user_question},
                {"answer": answer}
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            st.error(f"Something went wrong: {e}")