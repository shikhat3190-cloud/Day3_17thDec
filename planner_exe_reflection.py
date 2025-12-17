# Planner–Executor + Self-Reflection Loop
# Before: Planner → Executor → Output

from typing import List, Dict, Any
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv

load_dotenv()

# Schemas (Governance Layer)
class PlanStep(BaseModel):
    step_id: int
    action: str
    tool: str
    expected_output: str

class ExecutionPlan(BaseModel):
    objective: str
    steps: List[PlanStep]
    output_artifact: str

# Reflection Schema (Quality Gate)
class ReflectionResult(BaseModel):
    completeness_score: int   # 1–5
    clarity_score: int        # 1–5
    actionability_score: int  # 1–5
    format_compliance: bool
    decision: str             # accept | retry | escalate


planner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Planning Agent.\n"
     "Create a step-by-step execution plan.\n\n"
     "Rules:\n"
     "- Do NOT execute tools\n"
     "- Do NOT explain reasoning\n"
     "- Use only these tools: web_search, llm_analysis, llm_generation\n"
     "- Specify a markdown output file\n"
     "- Output must strictly match the JSON schema"),
    ("human", "{objective}")
])

planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
planner_chain = planner_prompt | planner_llm.with_structured_output(ExecutionPlan)

search_tool = TavilySearchResults(max_results=5)
executor_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def execute_plan(plan: ExecutionPlan) -> Dict[str, Any]:
    state: Dict[str, Any] = {}

    for step in plan.steps:
        print(f"Executing Step {step.step_id}: {step.action}")

        if step.tool == "web_search":
            state["search_results"] = search_tool.invoke(step.action)

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
                "- Clear headers\n"
                "- Enterprise focus\n"
                "- Concise and factual\n\n"
                f"Analysis:\n{state.get('analysis')}"
            )
            state["final_output"] = executor_llm.invoke(prompt).content

        else:
            raise ValueError(f"Unsupported tool: {step.tool}")

    return state

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Quality Evaluation Agent.\n"
     "Evaluate the document strictly against the criteria.\n"
     "Do NOT rewrite the content.\n"
     "Do NOT explain reasoning.\n"
     "Return structured evaluation only."),
    ("human",
     "Document:\n{document}\n\n"
     "Evaluation criteria:\n"
     "- Completeness (1–5)\n"
     "- Clarity (1–5)\n"
     "- Actionability (1–5)\n"
     "- Markdown format compliance (true/false)\n"
     "- Final decision: accept, retry, or escalate")
])

reflection_llm = ChatOpenAI(model="gpt-4o", temperature=0)
reflection_chain = reflection_prompt | reflection_llm.with_structured_output(ReflectionResult)

def reflect_and_decide(content: str) -> ReflectionResult:
    reflection = reflection_chain.invoke({"document": content})
    print("\nReflection Result:")
    print(reflection.model_dump_json(indent=2))
    return reflection

def write_markdown_file(filename: str, content: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    objective = (
        "Analyze recent Microsoft Copilot announcements "
        "and summarize enterprise impact for senior leadership."
    )

    # 1. Planning
    plan = planner_chain.invoke({"objective": objective})
    print("\n=== GENERATED PLAN ===")
    print(plan.model_dump_json(indent=2))

    MAX_RETRIES = 1
    attempt = 0

    while True:
        attempt += 1
        print(f"\n=== EXECUTION ATTEMPT {attempt} ===")

        # 2. Execution
        results = execute_plan(plan)

        # 3. Reflection
        reflection = reflect_and_decide(results["final_output"])

        # 4. Control Decision
        if reflection.decision == "accept":
            write_markdown_file(plan.output_artifact, results["final_output"])
            print(f"\nOutput accepted. Markdown file written: {plan.output_artifact}")
            break

        elif reflection.decision == "retry" and attempt <= MAX_RETRIES:
            print("\nRetrying execution with same plan...")
            continue

        else:
            print("\nEscalating to human review.")
            break

