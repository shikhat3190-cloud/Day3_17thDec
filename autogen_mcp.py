import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from dotenv import load_dotenv
load_dotenv()

async def main():
    # Get the fetch tool from mcp-server-fetch
    """
    What is mcp-server-fetch?
    mcp-server-fetch is an MCP tool server that:
    Accepts requests via stdin/stdout
    Fetches URLs over HTTP
    Returns page content to the agent
    Runs outside the Python process (sandboxed)
    It provides the “fetch” tool to your AutoGen agent.

    args=["mcp-server-filesystem"]
    args=["mcp-server-git"]
    args=["mcp-server-sqlite"]

    uvx is a command runner provided by uv (a fast Python package & environment manager).
    uvx mcp-server-fetch -
     - uvx checks if mcp-server-fetch is available
     - If not, it downloads required packages
     - Creates a temporary environment
     - Runs mcp-server-fetch
     mcp-server-fetch is a tool, and more specifically:
     an MCP tool server that provides a “fetch” capability to AI agents.
     What tool does it provide?
        fetch tool
        The mcp-server-fetch tool lets an agent:
        Fetch web pages
        Retrieve raw HTML/text
        Follow redirects
        Respect timeouts
    """
    fetch_mcp_server = StdioServerParams(
        command="uvx",   
        args=["mcp-server-fetch"],
        read_timeout_seconds=30.0,  # ⬅ CRITICAL FIX
    )

    # Create an MCP workbench session
    async with McpWorkbench(fetch_mcp_server) as workbench:  # type: ignore

        # Create model client
        model_client = OpenAIChatCompletionClient(
            model="gpt-4.1-nano"
        )
        # Create an agent that can use MCP fetch tool
        fetch_agent = AssistantAgent(
            name="fetcher",
            model_client=model_client,
            workbench=workbench,
            reflect_on_tool_use=True,
        )
        # Run the agent
        result = await fetch_agent.run(
            task="Summarize the content of https://en.wikipedia.org/wiki/Seattle"
        )
        # Print the final message
        assert isinstance(result.messages[-1], TextMessage)
        print(result.messages[-1].content)

        # Close model client
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
