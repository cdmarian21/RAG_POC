import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PDF_DIRECTORY = "./data" 
SLM_MODEL = "gemma4:e2b"  
#EMBEDDING_MODEL = "all-minilm"
EMBEDDING_MODEL = "mxbai-embed-large"

def main():
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    vector_store = None
    start_index = 0
    for i, chunk in enumerate(chunks):
        try:
            vector_store = FAISS.from_documents([chunk], embeddings)
            start_index = i + 1
            break
        except Exception as e:
            print(f"skipped chunk {i+1} - memory limit reached")

    successful_chunks = 1
    for i in range(start_index, len(chunks)):
        try:
            vector_store.add_documents([chunks[i]])
            successful_chunks += 1
            if successful_chunks % 20 == 0:
                print(f"  ... Embedded {successful_chunks} valid chunks")
        except Exception as e:
            print(f"skipped chunk {i+1} - memory limit reached")

    print(f"embedded {successful_chunks} out of {len(chunks)}")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = OllamaLLM(model=SLM_MODEL)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer concise and relevant.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm_chain = prompt | llm | StrOutputParser()

    
    while True:
        user_query = input("\nAsk a question about your PDFs (or type 'q' to quit): ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        print("...")
        
        retrieved_docs = retriever.invoke(user_query)
        
        formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        answer = llm_chain.invoke({
            "context": formatted_context,
            "input": user_query
        })
        
        print("\nAnswer:")
        print(answer)
        
        print("\nused:")
        for doc in retrieved_docs:
            print(f"- Page {doc.metadata.get('page', 'Unknown')} of {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()