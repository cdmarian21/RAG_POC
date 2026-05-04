from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config

def setup_pipeline():
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    
    # load the FAISS db
    try:
        vector_store = FAISS.load_local(
            folder_path=config.DB_PATH, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True        # have to allow this (make sure the files you are putting in are safe with no hidden code)
        )
    except Exception as e:
        print(f"Error loading db: {e}")
        return None, None, None

    if hasattr(vector_store.index, "nprobe"):
        vector_store.index.nprobe = config.NPROBE        # x closest clusters for better accuracy


    print("Building index")
    section_index = {}
    
    # creating a hash map to not have to re-organize the db for every question
    # technically not supposed to use _dict (protected attr) but need to in this case
    # if it fails in the future this is the first thing to troubleshoot
    for chunk in vector_store.docstore._dict.values():
        h1 = chunk.metadata.get('Header 1', '')
        h2 = chunk.metadata.get('Header 2', '')
        h3 = chunk.metadata.get('Header 3', '')
        source = chunk.metadata.get('source', 'unknown')
        
        # iteration for grouped chunks based on md headers
        if h1 or h2 or h3:
            section_id = f"{source}::{h1}::{h2}::{h3}"
            
            # create an empty list if this is the first time seeing the section
            if section_id not in section_index:
                section_index[section_id] = []
                
            # adding the chunk text to the section list
            section_index[section_id].append(chunk.page_content)
    

    retriever = vector_store.as_retriever(search_kwargs={"k": config.KWARGS})
    llm = OllamaLLM(model=config.SLM_MODEL)
    
    # prompt, change if needed
    system_prompt = (
        "You are an expert assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, say that you don't know. "
        "Keep the answer concise and well-structured.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm_chain = prompt | llm | StrOutputParser()        # LCEL pipeline
    return retriever, llm_chain, section_index

def ask_question(user_query, retriever, llm_chain, section_index):
    retrieved_docs = retriever.invoke(user_query)       # grabbing the 4 most relevant chunks

    assembled_context = []
    seen_sections = [] 
    
    # iterating through the likely docs to gain context
    for doc in retrieved_docs:
        source = doc.metadata.get('source', 'unknown')
        h1 = doc.metadata.get('Header 1', '')
        h2 = doc.metadata.get('Header 2', '')
        h3 = doc.metadata.get('Header 3', '')
        
        section_id = f"{source}::{h1}::{h2}::{h3}"
        
        # restiching the sections for better context
        if (h1 or h2 or h3) and section_id not in seen_sections:
            seen_sections.append(section_id) 
            
            section_chunks = section_index.get(section_id, [])
            
            header_title = h3 or h2 or h1
            stitched_text = f"Section: {header_title} (Source: {source})\n" + "\n".join(section_chunks)       # breadcrumb trail
            assembled_context.append(stitched_text)
            
        # failsafe for sections without headers
        elif not (h1 or h2 or h3):
            stitched_text = f"Section: Body Text (Source: {source})\n{doc.page_content}"                      # breadcrumb trail
            assembled_context.append(stitched_text)
            
            # adding to the list if it is not already there
            body_id = f"{source}::Body Text"
            if body_id not in seen_sections:
                seen_sections.append(body_id)
                
    formatted_context = "\n\n*****************\n\n".join(assembled_context)
    
    answer = llm_chain.invoke({
        "context": formatted_context,
        "input": user_query
    })
    return answer, seen_sections

def main():
    retriever, llm_chain, section_index = setup_pipeline()
    if not retriever:
        return
        
    while True:
        user_query = input("\nAsk a question about your files (or type 'quit'): ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        print("...")
        
        answer, seen_sections = ask_question(user_query, retriever, llm_chain, section_index)
        
        print("\nAnswer:")
        print(answer)
        
        # printing out the sections used so I can verify accuracy
        print("\nSections used:")
        for section in seen_sections:
            clean_source = " > ".join(part for part in section.split("::") if part)
            print(f"- {clean_source}")
            
if __name__ == "__main__":
    main()