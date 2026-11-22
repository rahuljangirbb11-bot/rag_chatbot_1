from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)


# ==========================================
# 1. CONTEXTUALIZATION PROMPT
# ==========================================

contextualize_prompt = ChatPromptTemplate.from_messages([

    (
        "system",

        """
        Given the conversation history and the latest
        user question, rewrite the question so that
        it can be understood independently.

        Do not answer the question.

        If the question is already understandable,
        return it unchanged.
        """
    ),

    MessagesPlaceholder("chat_history"),

    ("human", "{input}")

])


# ==========================================
# 2. QUESTION ANSWERING PROMPT
# ==========================================

qa_prompt = ChatPromptTemplate.from_messages([

    (
        "system",

        """
        You are an intelligent PDF-based AI assistant.

        Answer the user's question using ONLY the
        information provided in the retrieved context.

        Instructions:

        1. Give accurate and relevant answers totally based on the information provided in the book.

        2. If the answer is unavailable in the
        retrieved context, clearly say that you
        cannot find the information in the PDF.

        3. Do not invent facts.

        4. Explain complex concepts in simple language.

        5. Use bullet points wherever appropriate.

        6. Maintain conversational context.

        7. If appropriate, mention the relevant
        page numbers from the retrieved documents.

        8. Give answer with the classification provided in the book for relevance.

        Retrieved Context:

        {context}
        """
    ),

    MessagesPlaceholder("chat_history"),

    ("human", "{input}")

])