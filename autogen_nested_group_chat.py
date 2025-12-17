import asyncio

from autogen_agentchat.agents import UserProxyAgent, AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
load_dotenv()

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    writer = AssistantAgent(
        "writer", model_client=model_client, system_message="You are a writer.", model_client_stream=True
    )

    reviewer = AssistantAgent(
        "reviewer",
        model_client=model_client,
        system_message="Provide feedback to the input and suggest improvements.",
        model_client_stream=True,
    )

    # NOTE: you can skip input by pressing Enter.
    user_proxy = UserProxyAgent("user_proxy")

    # Maximum 1 round of review and revision.
    inner_termination = MaxMessageTermination(max_messages=4)

    # The outter-loop termination condition that will terminate the team when the user types "exit".
    outter_termination = TextMentionTermination("exit", sources=["user_proxy"])

    team = RoundRobinGroupChat(
        [
            # For each turn, the writer writes a summary and the reviewer reviews it.
            RoundRobinGroupChat([writer, reviewer], termination_condition=inner_termination),
            # The user proxy gets user input once the writer and reviewer have finished their actions.
            user_proxy,
        ],
        termination_condition=outter_termination,
    )
    # Start the team and wait for it to terminate.
    await Console(team.run_stream(task="Write a short essay about the impact of AI on society."))


asyncio.run(main())
