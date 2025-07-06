from typing import Annotated, Literal, TypedDict, Optional, List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from prompts import document_agent_system_prompt, financial_agent_system_prompt
from tools import doc_agent_tools, financial_agent_tools

load_dotenv()

all_tools = financial_agent_tools + doc_agent_tools
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash").bind_tools(all_tools)
financial_agent = create_react_agent(
    model="google_genai:gemini-2.0-flash",
    prompt=financial_agent_system_prompt,
    tools=financial_agent_tools,
    tool_choice="any",
)
doc_agent = create_react_agent(
    model="google_genai:gemini-2.0-flash",
    prompt=document_agent_system_prompt,
    tools=doc_agent_tools,
    tool_choice="any",

)

class State(TypedDict):
    query: str
    document_sources: Optional[List[str]]
    collection: str
    symbol: str
    messages: Annotated[list, add_messages]


def router(state: State) -> State:
    if len(state["document_sources"]):
        return {"next": "DocumentAgent"}
    else:
        return {"next": "FinancialAgent"}


def FinancialAgent(state: State) -> State:
    last_message = state["messages"][-1]
    messages = [
        {"role": "system", "content": financial_agent_system_prompt},
        {
            "role": "user",
            "content": last_message.content
            + "\n\n"
            + f"Provided Ticker Symbol {state["symbol"]}\n\n",
        },
    ]
    response = financial_agent.invoke({"messages":messages})
    content = response["messages"][-1].content
    return {"messages": [{"role": "assistant", "content": content}]}


def DocumentAgent(state: State) -> State:
    last_message = state["messages"][-1] 
    system_message = SystemMessage(content=f"""{document_agent_system_prompt}
    Available Context:
    - Ticker Symbol: {state["symbol"]}
    - Collection: {state["collection"]}
    - Document Sources: {[doc for doc in state["document_sources"]]}

    Use the `retrieve_from_documents` tool to search for relevant information based on the user's query.""")
    conversation_messages = [system_message]
    conversation_messages.append(last_message)


    response = doc_agent.invoke({"messages":conversation_messages})
    content = response["messages"][-1].content
    return {"messages": [{"role": "assistant", "content": content}]}


def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


financial_agent_tool_node = ToolNode(financial_agent_tools)
doc_agent_tool_node = ToolNode(doc_agent_tools)
graph = StateGraph(State)
graph.add_node("FinancialAgent", FinancialAgent)
graph.add_node("DocumentAgent", DocumentAgent)
graph.add_node(
    "financial_agent_tool_node", financial_agent_tool_node
)
graph.add_node("doc_agent_tools", doc_agent_tool_node)
graph.add_node("router", router)
graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {"DocumentAgent": "DocumentAgent", "FinancialAgent": "FinancialAgent"},
)

graph.add_conditional_edges(
    "FinancialAgent",
    should_continue,
    {"continue": "financial_agent_tool_node", "end": END},
)

graph.add_conditional_edges(
    "DocumentAgent", should_continue, {"continue": "doc_agent_tools", "end": END}
)

graph.add_edge("financial_agent_tool_node", "FinancialAgent")
graph.add_edge("doc_agent_tools", "DocumentAgent")

app = graph.compile()
# png = app.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png)


def run_chatbot(query: str, symbol: str, document_sources: list[str], collection: str) -> dict:
    inputs = {
        "query": query,
        "symbol": symbol,
        "document_sources": document_sources,
        "collection": collection,
        "messages": [{"role": "user", "content": query}],
    }
    response = app.invoke(inputs)
    return response["messages"][-1].content