from typing import TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph
from dotenv import load_dotenv

load_dotenv()
# AgentState is the shared memory of your agent graph.
class AgentState(TypedDict):
    objective: str
    plan: Optional[dict]
    search_results: Optional[list]
    analysis: Optional[str]
    final_output: Optional[str]
    reflection: Optional[dict]
    retry_count: int

from pydantic import BaseModel

class PlanStep(BaseModel):
    step_id: int
    action: str
    tool: str
    expected_output: str

class ExecutionPlan(BaseModel):
    objective: str
    steps: List[PlanStep]
    output_artifact: str

class ReflectionResult(BaseModel):
    completeness_score: int
    clarity_score: int
    actionability_score: int
    format_compliance: bool
    decision: str  # accept | retry | escalate



planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
executor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
reflection_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

search_tool = TavilySearchResults(max_results=5)

# 6. Nodes (LangGraph Core)

## Planner Node
from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Planning Agent.\n"
     "Create a step-by-step execution plan.\n"
     "Rules:\n"
     "- Do NOT execute tools\n"
     "- Do NOT explain reasoning\n"
     "- Use only tools: web_search, llm_analysis, llm_generation\n"
     "- Specify markdown output file\n"
     "- Output must match schema"),
    ("human", "{objective}")
])

planner_chain = planner_prompt | planner_llm.with_structured_output(ExecutionPlan)

def planner_node(state: AgentState):
    plan = planner_chain.invoke({"objective": state["objective"]})
    state["plan"] = plan.model_dump()
    return state

## Executor Node
def executor_node(state: AgentState):
    for step in state["plan"]["steps"]:
        if step["tool"] == "web_search":
            state["search_results"] = search_tool.invoke(step["action"])

        elif step["tool"] == "llm_analysis":
            prompt = f"Extract enterprise insights:\n{state['search_results']}"
            state["analysis"] = executor_llm.invoke(prompt).content

        elif step["tool"] == "llm_generation":
            prompt = (
                "Create executive-ready Markdown summary.\n\n"
                f"Analysis:\n{state['analysis']}"
            )
            state["final_output"] = executor_llm.invoke(prompt).content

    return state

## Reflection Node
reflection_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Quality Evaluation Agent.\n"
     "Do NOT rewrite content.\n"
     "Do NOT explain reasoning.\n"
     "Return structured evaluation only."),
    ("human",
     "Document:\n{document}\n\n"
     "Evaluate completeness, clarity, actionability, format.\n"
     "Return accept, retry, or escalate.")
])

reflection_chain = reflection_prompt | reflection_llm.with_structured_output(ReflectionResult)

def reflection_node(state: AgentState):
    reflection = reflection_chain.invoke({
        "document": state["final_output"]
    })
    state["reflection"] = reflection.model_dump()
    return state

MAX_RETRIES = 1

def decision_router(state: AgentState):
    decision = state["reflection"]["decision"]

    if decision == "accept":
        return "persist"

    if decision == "retry" and state["retry_count"] < MAX_RETRIES:
        state["retry_count"] += 1
        return "execute"

    return "interrupt"

def persist_node(state: AgentState):
    filename = state["plan"]["output_artifact"]
    with open(filename, "w", encoding="utf-8") as f:
        f.write(state["final_output"])
    return state

def interrupt_node(state: AgentState):
    raise RuntimeError("Escalated for human review")



graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("execute", executor_node)
graph.add_node("reflect", reflection_node)
graph.add_node("persist", persist_node)
graph.add_node("interrupt", interrupt_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "execute")
graph.add_edge("execute", "reflect")

graph.add_conditional_edges(
    "reflect",
    decision_router,
    {
        "persist": "persist",
        "execute": "execute",
        "interrupt": "interrupt"
    }
)

agent = graph.compile()

result = agent.invoke({
    "objective": "Analyze recent Microsoft Copilot announcements and summarize enterprise impact",
    "retry_count": 0
})
print(result)
