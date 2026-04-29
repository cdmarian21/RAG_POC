from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import config

# for testing purposes. Needed to identify which chunks are seperated/causing loss of context
def peek_at_database():
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        folder_path=config.DB_PATH, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )

    all_documents = vector_store.docstore._dict
    print(f"Total chunks in database: {len(all_documents)}\n")
    
    for i, (doc_id, document) in enumerate(list(all_documents.items())[:50]):
        print(f"=== Chunk {i+1} ===")
        print(f"Internal ID: {doc_id}")
        print(f"Metadata: {document.metadata}")
        print(f"Content preview: {document.page_content[:700]}...\n")

if __name__ == "__main__":
    peek_at_database()