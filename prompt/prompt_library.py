from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt for document analysis
document_analysis_prompt = ChatPromptTemplate.from_template(
        ''' You are a highly capable assistant specialized in **document analysis and summarization**.  
            Your output must be **only valid JSON** that strictly conforms to the schema provided below.  

            ⚠️ Do not include explanations, reasoning, or any extra text outside of the JSON object.  
            ⚠️ Do not modify, remove, or reorder any schema fields.  
            ⚠️ Ensure every value matches the expected type in the schema.  

            Schema:
            {format_instructions}

            Task:
            1. Carefully read and analyze the following document.  
            2. Extract, summarize, or transform its content according to the schema requirements.  
            3. Return your response as a **single JSON object** that strictly follows the schema.  

            Document to analyze:  
            {document_text} '''
)

# Prompt for document comparison
document_comparison_prompt = ChatPromptTemplate.from_template(
        ''' You are a highly capable assistant specialized in **document comparison and structured reporting**.  
            You will be provided with the combined content from two PDFs.  

            ⚠️ Your task is to produce a **page-wise comparison**. 
            ⚠️ Ensure every page number is preserved in the output, even if no changes exist.  

            ### Tasks:
            1. Compare the content in the two PDFs page by page.  
            2. For each page:  
            - If differences exist, explicitly list them.  
            - If no differences exist, record `"NO CHANGE"`.  
            3. Ensure the output is structured **exactly** as per the schema.  

            ### Input Documents:
            {combined_docs}

            ### Output Format:
            {format_instruction} '''
)

# Prompt for contextual question rewriting
contextualize_question_prompt = ChatPromptTemplate.from_messages([
        ("system", 
            "You are a highly capable assistant specialized in query reformulation. "
            "Your task is to take the given conversation history along with the most recent user query "
            "and rewrite the query into a fully standalone version that can be understood without any prior context. "

            "⚠️ Important Rules: "
            "1. Do NOT provide an answer—only return the reformulated query. "
            "2. If the query already makes complete sense on its own, return it unchanged. "
            "3. Preserve the original intent, tone, and meaning exactly. "
            "4. Ensure clarity and grammatical correctness in the rewritten query. "
            "5. Do not add, infer, or hallucinate extra details not present in the query or conversation history."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
])

# Prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
            "You are an assistant designed to answer questions strictly based on the provided context. "
            "Rules you must follow: "
                "1. Use only the retrieved context to generate your response. "
                "2. If the answer is not explicitly present in the context, reply with exactly: 'I don't know.' "
                "3. Keep responses clear, factual, and no longer than three sentences. "
                "4. Do not include assumptions, outside knowledge, or fabricated details. "
                "5. Maintain a neutral, professional tone. "
            "\n\nContext:\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
])

# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}