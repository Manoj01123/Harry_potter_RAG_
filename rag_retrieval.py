
from langchain.llms import LlamaCpp
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import time
import logging

# Configure logging to track metrics in a separate file
logging.basicConfig(
    filename="metric_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Helper function to truncate documents
def truncate_documents(documents, token_limit=512):
    truncated_docs = []
    for doc in documents:
        if len(doc.page_content) > token_limit:
            truncated_docs.append(doc.page_content[:token_limit] + "...")
        else:
            truncated_docs.append(doc.page_content)
    return truncated_docs

# Helper function to count tokens
def count_tokens(text):
    splitter = TokenTextSplitter(chunk_size=1024)
    return len(splitter.split_text(text))

# Load Hugging FaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = FastTextEmbeddings(file_path="cc.en.300.vec")  # Path to pretrained embeddings

# Load ChromaDB (ensure it's already populated)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Test retrieval logic
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Retrieve fewer docs

# Load the GGUF model with LlamaCpp
llm = LlamaCpp(model_path="./models/Llama-3.2-3B-Instruct-Q4_0.gguf", n_ctx=2048)

# Define the Retrieval-Augmented Generation pipeline
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


def query_rag(question):
    start_time = time.time()

    # Retrieve documents
    retrieval_start = time.time()
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieval_time = time.time() - retrieval_start

    truncated_docs = truncate_documents(retrieved_docs, token_limit=512)  # Truncate long docs

    # Log token counts for debugging
    total_tokens = count_tokens(question + " ".join(truncated_docs))

    # Query the RAG chain
    rag_start = time.time()
    response = rag_chain({"query": question, "retrieved_documents": truncated_docs})
    rag_time = time.time() - rag_start

    total_time = time.time() - start_time

    # Log metrics to a file

    logging.info(f"Question: {question}")
    logging.info(f"Answer: {response['result']}")
    logging.info(f"Total Tokens Processed: {total_tokens}")
    logging.info(f"Retrieval Time: {retrieval_time:.4f} seconds")
    logging.info(f"RAG Processing Time: {rag_time:.4f} seconds")
    logging.info(f"Total Query Time: {total_time:.4f} seconds")

    return response

# Main execution
if __name__ == "__main__":
    print("Welcome to the interactive Harry Potter Q&A system!")
    print("Type your questions below. Type 'exit' to quit the application.")

    while True:
        question = input("Question: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Get the response from the RAG system
        answer = query_rag(question)
        print("\nAnswer:", answer["result"])

        # Display source documents
        # print("\nSource Documents:")
        # for i, doc in enumerate(answer["source_documents"], 1):
        #     print(f"Document {i}: {doc.page_content[:200]}...")
