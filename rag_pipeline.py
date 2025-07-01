import os
import glob
from typing import TypedDict

from langchain_community.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda

from dotenv import load_dotenv
import os

load_dotenv()  # ✅ This loads .env variables into os.environ

openai_api_key = os.getenv("OPENAI_API_KEY")  # ✅ Gets your key safely



# 📥 Load PDF documents
loaders = [PyPDFLoader(pdf) for pdf in glob.glob("data/*.pdf")]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# ✂️ Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# 🧠 Generate embeddings and store in FAISS
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever()

# 💬 Define LangGraph state
from typing import List, TypedDict, Any

class GraphState(TypedDict):
    question: str
    answer: str
    chat_history: List[dict]
    docs: Any  # 🧠 or List[Document] if you want it strongly typed


# 🔍 Step 1: Retrieve relevant documents
def retrieve(state: GraphState):
    return {
        "question": state["question"],
        "chat_history": state["chat_history"],
        "docs": retriever.get_relevant_documents(state["question"])  # ✅ MUST BE PRESENT
    }


# 🧠 Step 2: Generate answer using context + chat history
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following chat history and context to answer the user's latest question.

Chat History:
{chat_history}

Context:
{context}

Question: {question}
""")

def generate(state):
    context = "\n\n".join([doc.page_content for doc in state["docs"]])
    history_text = "\n".join(
        [f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in state["chat_history"]]
    )

    final_prompt = prompt.format(context=context, chat_history=history_text, question=state["question"])
    answer = llm.invoke(final_prompt).content

    return {
        "question": state["question"],
        "answer": answer,
        "chat_history": state["chat_history"] + [{"question": state["question"], "answer": answer}],
        "docs": state["docs"]  # ✅ fix: return 'docs' so next step can use it
    }



# ✅ Step 3: Confidence check
def check_confidence(state):
    decision = "retry" if "not confident" in state["answer"].lower() else "end"
    return {**state, "next": decision}  # 👈 ✅ return full state with a 'next' key


# 🔄 LangGraph setup
graph = StateGraph(GraphState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("check", RunnableLambda(check_confidence))

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "check")
graph.add_conditional_edges(
    "check",
    lambda s: s["next"],  # 👈 extract 'next' key from state
    {
        "retry": "retrieve",
        "end": END
    }
)


app = graph.compile()

# ❓ Ask a question
initial_state = {
    "question": "what are the newe products that came out? ",
    "chat_history": []
    # ❌ Do not include 'docs'
}


result = app.invoke(initial_state)

print("\n🧠 Final Answer:\n", result["answer"])
