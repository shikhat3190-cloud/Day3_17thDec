#  Planner → Executor → Output

from typing import List, Dict, Any
from pydantic import BaseModel

from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv

load_dotenv()

"""    
################################################
Define the Planning Schema (Governance Layer)
This is critical for enterprise explainability
################################################
"""
class PlanStep(BaseModel):
    step_id: int
    action: str
    tool: str
    expected_output: str

class ExecutionPlan(BaseModel):
    objective: str
    steps: List[PlanStep]
    output_artifact: str  # e.g. "executive_summary.md"


from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Planning Agent.\n"
     "Create a step-by-step execution plan.\n\n"
     "Rules:\n"
     "- Do NOT execute tools\n"
     "- Do NOT explain reasoning\n"
     "- Use only the following tools: web_search, llm_analysis, llm_generation\n"
     "- Specify a markdown output file name\n"
     "- Output must strictly match the JSON schema\n"),
    ("human", "{objective}")
])


from langchain_openai import ChatOpenAI

planner_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

planner_chain = planner_prompt | planner_llm.with_structured_output(ExecutionPlan)


planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)

planner_chain = planner_prompt | planner_llm.with_structured_output(ExecutionPlan)

objective = "Analyze Microsoft Copilot recent announcements and summarize enterprise impact."

plan = planner_chain.invoke({"objective": objective})

search_tool = TavilySearchResults(max_results=5)
executor_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def execute_plan(plan: ExecutionPlan) -> Dict[str, Any]:
    state: Dict[str, Any] = {}

    for step in plan.steps:
        print(f"\nExecuting Step {step.step_id}: {step.action}")

        if step.tool == "web_search":
            result = search_tool.invoke(step.action)
            state["search_results"] = result

        elif step.tool == "llm_analysis":
            prompt = (
                "Extract enterprise-relevant insights from the following data:\n\n"
                f"{state.get('search_results')}"
            )
            state["analysis"] = executor_llm.invoke(prompt).content


        elif step.tool == "llm_generation":
            prompt = (
                "Create an executive-ready summary in Markdown format.\n\n"
                "Requirements:\n"
                "- Use clear section headers\n"
                "- Focus on enterprise impact\n"
                "- Be concise and factual\n\n"
                f"Analysis:\n{state.get('analysis')}"
            )
            state["final_output"] = executor_llm.invoke(prompt).content
        else:
            raise ValueError(f"Unsupported tool: {step.tool}")

    return state

def write_markdown_file(filename: str, content: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    objective = (
        "Analyze recent Microsoft Copilot announcements "
        "and summarize enterprise impact for senior leadership."
    )

    # Planning phase
    plan = planner_chain.invoke({"objective": objective})

    print("\n=== GENERATED PLAN ===")
    print(plan.model_dump_json(indent=2))

    # Execution phase
    results = execute_plan(plan)

    # Persist artifact
    write_markdown_file(
        plan.output_artifact,
        results["final_output"]
    )

    print(f"\nMarkdown report generated: {plan.output_artifact}")
