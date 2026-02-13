"""
LangGraph-based wrapper around the existing RAG pipeline.

One-node graph:
- Input: user question as a HumanMessage.
- Node: calls your RAG functions and returns an AIMessage answer.
"""

# pip install langgraph

from typing import Annotated

from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from src.rag_query import load_index, retrieve_top_k, build_context, answer_query
from src.settings import get_setting, get_active_profile


# --- State definition -------------------------------------------------------

class State(TypedDict):
    # Messages list as state; add_messages appends new messages
    messages: Annotated[list, add_messages]


# --- Node: RAG answer generator --------------------------------------------

def rag_node(state: State) -> dict:
    """
    Node that takes the latest human message, runs RAG, and appends an AI answer.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    # Last message should be the user question
    last = messages[-1]
    if isinstance(last, HumanMessage):
        question = last.content
    else:
        # If last message isn't human, do nothing
        return {}

    # Load index for active profile
    profile = get_active_profile()
    print(f"[LangGraph] Active profile: {profile}")

    try:
        df_index, emb_array = load_index()
    except FileNotFoundError:
        answer_text = (
            "I don't see any index for the current profile. "
            "Please run ingestion before asking questions."
        )
        return {"messages": [AIMessage(content=answer_text)]}

    top_k = int(get_setting("rag.top_k", 5))

    # Standard RAG pipeline
    chunks = retrieve_top_k(df_index, emb_array, question, k=top_k)
    context = build_context(chunks)
    answer = answer_query(question, context)

    return {"messages": [AIMessage(content=answer)]}


# --- Build and compile graph -----------------------------------------------

def build_graph():
    graph_builder = StateGraph(State)

    graph_builder.add_node("rag_node", rag_node)

    graph_builder.add_edge(START, "rag_node")
    graph_builder.add_edge("rag_node", END)

    graph = graph_builder.compile()
    return graph


def run_once(question: str) -> str:
    """
    Convenience function: run the graph once for a single question.
    """
    graph = build_graph()

    result_state = graph.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    messages = result_state.get("messages", [])
    # Last message should be the AI answer
    if messages and isinstance(messages[-1], AIMessage):
        return messages[-1].content
    elif messages:
        return str(messages[-1].content)
    else:
        return ""


if __name__ == "__main__":
    print("LangGraph RAG agent - single turn demo")
    while True:
        q = input("Question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        ans = run_once(q)
        print("ANSWER:", ans)
        print("-" * 80)
