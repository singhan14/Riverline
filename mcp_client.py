import asyncio
import os
import sys
from typing import List, Optional

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Helper to run async code in sync context (for Streamlit/LangGraph)
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        # If loop is running (e.g. in Streamlit sometimes), use future
        import concurrent.futures
        pool = concurrent.futures.ThreadPoolExecutor()
        return pool.submit(asyncio.run, coro).result()
    else:
        return loop.run_until_complete(coro)

class RiverlineMCPClient:
    def __init__(self):
        # Resolve absolute path to server script
        server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
        
        self.server_params = StdioServerParameters(
            command=sys.executable, # Use the current python interpreter
            args=[server_script],
            env=os.environ.copy()
        )
    
    async def _get_tools_async(self):
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List tools from the server
                result = await session.list_tools()
                
                langchain_tools = []
                for tool in result.tools:
                    # Capture variable in closure for the tool function
                    # We need to create a standalone function that calls the MCP server
                    # NOTE: This is tricky because we need a Persistent connection for the tool call
                    # For a simple demo, we will re-connect on every call (Inefficient but robust for scripts)
                    # A better way is a long-running server, but let's stick to simple.
                    
                    def make_tool_func(tool_name):
                        async def _tool_wrapper(**kwargs):
                            async with stdio_client(self.server_params) as (read, write):
                                async with ClientSession(read, write) as session:
                                    await session.initialize()
                                    res = await session.call_tool(tool_name, arguments=kwargs)
                                    # res is CallToolResult
                                    # Extract text
                                    output = []
                                    for content in res.content:
                                        if content.type == 'text':
                                            output.append(content.text)
                                    return "\n".join(output)
                        
                        def _sync_wrapper(**kwargs):
                           return run_async(_tool_wrapper(**kwargs))
                           
                        return _sync_wrapper

                    lc_tool = StructuredTool.from_function(
                        func=make_tool_func(tool.name),
                        name=tool.name,
                        description=tool.description or "MCP Tool"
                    )
                    langchain_tools.append(lc_tool)
                    
                return langchain_tools

    def get_tools(self):
        return run_async(self._get_tools_async())

# Global instance
mcp_client = RiverlineMCPClient()
