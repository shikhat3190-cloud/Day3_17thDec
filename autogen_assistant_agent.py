import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


# Define a tool that searches the web (mock)
async def web_search(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."


async def main():
    # Create the model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        # api_key="YOUR_API_KEY"
    )

    # Create the assistant agent with tool
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[web_search],
        system_message="Use tools to solve tasks.",
    )

    # Option1 : Run the agent normally
    result = await agent.run(
         task="Find information on AutoGen"
     )
     #Print result
    print(result.messages)

    # Option 2: use Console to print all messages as they appear.
    await Console(
        agent.run_stream(task="Find information on AutoGen"),
        output_stats=True,  # Enable stats printing.
    )

    # Close client
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
