from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load and split the PDF
loader = PyPDFLoader('paper.pdf')
pages = loader.load_and_split()
print(f"Loaded {len(pages)} pages from PDF")

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

# Create HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Created HuggingFace embeddings")

# Create FAISS vector store
vector_db = FAISS.from_documents(chunks, embeddings)
print("Created FAISS vector store")

# Set up the language model
local_model = "qwen2:1.5b"
llm = ChatOllama(model=local_model)

# Set up the retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# RAG prompt
template = """Answer the question based ONLY on the following context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

#Rules
 -Extract the 10 Most Important Key Points From the Paper.
 -Present Them In A very Good Way.
 -Do not Go out of Context and use your own knowledge Base.
 -Only Provide Authentic and Real Information From the Paper.
 -Present the Key points in Bullet Points

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Set up the chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Hardcoded questions
questions = [
    "What are the main findings of this paper?",
    "Summarize the methodology used in this research.",
    "What are the key contributions of this paper to the field?",
    "What are the limitations of this study?",
    "What future research directions are suggested by this paper?"
]

# Run the chain
try:
    print("Please select a question by entering its number:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    choice = int(input("Enter your choice (1-5): "))
    if 1 <= choice <= 5:
        question = questions[choice - 1]
    else:
        raise ValueError("Invalid choice. Please select a number between 1 and 5.")
    
    print(f"\nSelected Question: {question}\n")
    
    # Debug: Print retrieved documents
    retrieved_docs = retriever.get_relevant_documents(question)
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i + 1}:")
        print(doc.page_content[:200] + "...")  # Print first 200 characters
        print()

    response = chain.invoke(question)
    print("Response:")
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")