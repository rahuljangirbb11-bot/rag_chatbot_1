from langchain_openai import ChatOpenAI

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_classic.memory import ConversationBufferMemory

from config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    RETRIEVER_K
)

from prompts import (
    contextualize_prompt,
    qa_prompt
)


# ==========================================
# 1. INITIALIZE LANGUAGE MODEL
# ==========================================

def create_llm():

    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is missing.")

    llm = ChatOpenAI(

        model=LLM_MODEL,

        temperature=0.2,

        api_key=OPENAI_API_KEY

    )

    return llm


# ==========================================
# 2. CREATE CONVERSATIONAL MEMORY
# ==========================================

def create_memory():

    memory = ConversationBufferMemory(

        memory_key="chat_history",

        return_messages=True,

        input_key="input",

        output_key="answer"

    )

    return memory


# ==========================================
# 3. CREATE RAG CHAIN
# ==========================================

def create_rag_chain(vectorstore):

    # Initialize LLM

    llm = create_llm()


    # Create retriever

    retriever = vectorstore.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": RETRIEVER_K
        }

    )


    # Create history-aware retriever

    history_aware_retriever = create_history_aware_retriever(

        llm,

        retriever,

        contextualize_prompt

    )


    # Create document combination chain

    question_answer_chain = create_stuff_documents_chain(

        llm,

        qa_prompt

    )


    # Create final retrieval chain

    rag_chain = create_retrieval_chain(

        history_aware_retriever,

        question_answer_chain

    )


    return rag_chain