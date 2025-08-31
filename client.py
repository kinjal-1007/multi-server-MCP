import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")


class Server:
    """MCP server connection and tool execution."""
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock = asyncio.Lock()

    async def initialize(self) -> None:
        command = shutil.which(self.config["command"]) or self.config["command"]
        if not command:
            raise ValueError("Invalid command for server")
        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config.get("env", {})}
        )
        try:
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Failed to initialize {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        tools_response = await self.session.list_tools()
        tools = []
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(Tool(tool.name, tool.description, tool.inputSchema, tool.title) for tool in item[1])
        return tools

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], retries: int = 2, delay: float = 1.0) -> Any:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                return await self.session.call_tool(tool_name, arguments)
            except Exception as e:
                attempt += 1
                logging.warning(f"Attempt {attempt} failed: {e}")
                if attempt < retries:
                    await asyncio.sleep(delay)
                else:
                    raise

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
            except Exception as e:
                logging.error(f"Cleanup error for {self.name}: {e}")


class Tool:
    def __init__(self, name: str, description: str, input_schema: dict[str, Any], title: str | None = None):
        self.name = name
        self.title = title
        self.description = description
        self.input_schema = input_schema

    def format_for_llm(self) -> str:
        args_desc = []
        for param, info in self.input_schema.get("properties", {}).items():
            desc = f"- {param}: {info.get('description', 'No description')}"
            if param in self.input_schema.get("required", []):
                desc += " (required)"
            args_desc.append(desc)
        output = f"Tool: {self.name}\n"
        if self.title:
            output += f"User-readable title: {self.title}\n"
        output += f"Description: {self.description}\nArguments:\n{chr(10).join(args_desc)}\n"
        return output


class LLMClient:
    """Wrapper for Groq API."""
    def __init__(self, api_key: str, model: str = "qwen/qwen3-32b"):
        self.api_key = api_key
        self.model = model
        # Pass api_key when initializing ChatGroq
        self.client = ChatGroq(model=self.model, api_key=self.api_key)

    def get_response(self, messages: list[dict[str, str]]) -> str:
        # Convert dict messages to LangChain message objects
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        # Call the client with proper message objects
        response = self.client.invoke(langchain_messages)
        return response.content

class ChatSession:
    """Main chat session using MCP tools and LLM (Groq)."""
    def __init__(self, servers: list[Server], llm_client: LLMClient):
        self.servers = servers
        self.llm_client = llm_client

    def should_use_tool(self, user_input: str, llm_response: str) -> bool:
        """Determine if the query should have used a tool but didn't"""
        user_lower = user_input.lower()
        response_lower = llm_response.lower()
        
        # Keywords that suggest tool usage
        tool_keywords = [
            "restaurant", "nearby", "find", "search", "book", "reserve",
            "calendar", "schedule", "meeting", "appointment", "event",
            "leave", "vacation", "time off", "project", "task", "database", "query"
        ]
        
        # If user mentioned tool keywords but response doesn't contain JSON
        mentions_tool_concepts = any(keyword in user_lower for keyword in tool_keywords)
        response_not_json = not (response_lower.startswith('{') and response_lower.endswith('}'))
        
        return mentions_tool_concepts and response_not_json

    async def cleanup_servers(self):
        for server in reversed(self.servers):
            await server.cleanup()

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Execute a tool call and return (success, result)"""
        for server in self.servers:
            try:
                tools = await server.list_tools()
                if any(t.name == tool_name for t in tools):
                    logging.info(f"Executing tool: {tool_name} with arguments: {arguments}")
                    result = await server.execute_tool(tool_name, arguments)
                    return True, str(result)
            except Exception as e:
                logging.error(f"Error executing tool {tool_name}: {e}")
                return False, f"Error executing {tool_name}: {str(e)}"
        
        return False, f"Tool '{tool_name}' not found in any server"

    async def process_llm_response(self, llm_response: str) -> tuple[str, str | None]:
        """Process LLM response and execute tool if needed. Returns (display_text, tool_result)"""
        # Clean the response and extract content after </think> if present
        cleaned_response = llm_response.strip()
        
        # Check if response contains <think> tags and extract content after </think>
        if "</think>" in cleaned_response:
            # Split on </think> and take everything after it
            parts = cleaned_response.split("</think>", 1)
            if len(parts) > 1:
                # Get content after </think> and strip whitespace
                post_think_content = parts[1].strip()
                print(f"🧠 Extracted content after </think>: {post_think_content[:200]}...")
                cleaned_response = post_think_content
        
        # Try to parse as JSON tool call
        try:
            tool_call = json.loads(cleaned_response)
            
            if isinstance(tool_call, dict) and "tool" in tool_call and "arguments" in tool_call:
                tool_name = tool_call["tool"]
                arguments = tool_call["arguments"]
                
                print(f"🔧 Executing tool: {tool_name} with args: {arguments}")
                success, result = await self.execute_tool_call(tool_name, arguments)
                
                if success:
                    print(f"✅ Tool executed successfully")
                    return cleaned_response, result
                else:
                    print(f"❌ Tool execution failed: {result}")
                    return f"Tool execution failed: {result}", None
            else:
                # JSON but not a valid tool call format
                return cleaned_response, None
                
        except json.JSONDecodeError:
            # Check if response mentions needing to use tools but didn't format correctly
            if any(keyword in cleaned_response.lower() for keyword in ["restaurant", "nearby", "find", "search", "calendar", "book", "schedule"]):
                print("⚠️ LLM mentioned tool-related keywords but didn't format as JSON. Prompting for correct format...")
                return cleaned_response, None
            
            # Not JSON, treat as regular response
            return cleaned_response, None

    async def start(self):
        print("🚀 Initializing servers...")
        for server in self.servers:
            await server.initialize()
            print(f"✅ Server '{server.name}' initialized")

        # Collect all available tools
        all_tools = []
        for server in self.servers:
            tools = await server.list_tools()
            all_tools.extend(tools)
            print(f"📋 Server '{server.name}' has {len(tools)} tools")

        tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
        print(f"🛠️  Total tools available: {len(all_tools)}")

        from datetime import date
        today_str = date.today().strftime("%Y-%m-%d")
        system_message = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{tools_description}\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- When you need to use a tool, respond with ONLY the JSON object - no other text\n"
            "- Use this EXACT format for tool calls:\n"
            "{\n"
            '  "tool": "exact-tool-name",\n'
            '  "arguments": {"param1": "value1", "param2": "value2"}\n'
            "}\n\n"
            "- When you don't need tools, respond in natural language\n"
            "- For multi-step requests (like find restaurant AND book calendar), handle them step by step\n"
            "- After receiving tool results, you can make another tool call if needed\n"
            "- Always use tools when users ask for specific information that requires data lookup\n\n"
            "Examples of when to use tools:\n"
            "- Finding restaurants → use find_restaurants_nearby\n"
            "- Calendar/scheduling questions → use calendar tools\n"
            "- Leave management → use leave tools\n"
            "- Project management → use project tools\n"
            "- Database queries → use query tools\n\n"
            "For calendar bookings, use format: YYYY-MM-DDTHH:MM:SS and timezone like 'Asia/Kolkata'\n"
            f"Today's date for reference: {today_str}\n"
        )

        messages = [{"role": "system", "content": system_message}]
        print("\n💬 Chat started! Type 'quit' or 'exit' to end the session.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})
                
                # Get LLM response
                print("🤖 Assistant is thinking...")
                MAX_HISTORY = 8  # or 10
                llm_response = self.llm_client.get_response(messages[-MAX_HISTORY:])

                print(f"🔍 LLM Response: {llm_response[:200]}...")  # Debug: show first 200 chars
                
                # Process the response (check for tool calls)
                display_text, tool_result = await self.process_llm_response(llm_response)
                
                if tool_result is not None:
                    # Tool was executed, add both the tool call and result to conversation
                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "system", "content": f"Tool result: {tool_result}"})
                    
                    # Add instruction to show results and continue if needed
                    follow_up_instruction = (
                        "Based on the tool result above, provide a helpful response to the user. "
                        "If the user's original request requires additional steps (like booking a calendar event), "
                        "you can make another tool call by responding with JSON format again, "
                        "or provide the information and ask what they'd like to do next."
                    )
                    messages.append({"role": "system", "content": follow_up_instruction})
                    
                    # Get final natural language response
                    print("🤖 Processing tool results...")
                    final_response = self.llm_client.get_response(messages)
                    print(f"🔍 Final Response: {final_response[:200]}...")  # Debug final response
                    
                    # Check if the final response is another tool call
                    final_display, final_tool_result = await self.process_llm_response(final_response)
                    
                    if final_tool_result is not None:
                        # Another tool was called
                        print("🔄 Executing follow-up tool...")
                        messages.append({"role": "assistant", "content": final_response})
                        messages.append({"role": "system", "content": f"Tool result: {final_tool_result}"})
                        
                        # Get the truly final response
                        truly_final_response = self.llm_client.get_response(messages)
                        print(f"Assistant: {truly_final_response}")
                        messages.append({"role": "assistant", "content": truly_final_response})
                    else:
                        # Final response is natural language
                        print(f"Assistant: {final_display}")
                        messages.append({"role": "assistant", "content": final_display})
                else:
                    # Check if this looks like it should have been a tool call
                    if self.should_use_tool(user_input, display_text):
                        print("🔄 It looks like you need a tool. Let me try a more direct approach...")
                        # Add a more direct instruction
                        force_tool_message = (
                            f"The user asked: '{user_input}'. You have tools available that can help. "
                            "Please respond with the appropriate tool call in JSON format only."
                        )
                        messages.append({"role": "system", "content": force_tool_message})
                        
                        retry_response = self.llm_client.get_response(messages)
                        print(f"🔍 Retry Response: {retry_response[:200]}...")
                        
                        retry_display, retry_tool_result = await self.process_llm_response(retry_response)
                        
                        if retry_tool_result is not None:
                            # Tool worked on retry
                            messages.append({"role": "assistant", "content": retry_response})
                            messages.append({"role": "system", "content": f"Tool result: {retry_tool_result}"})
                            
                            final_response = self.llm_client.get_response(messages)
                            print(f"Assistant: {final_response}")
                            messages.append({"role": "assistant", "content": final_response})
                        else:
                            # Still no tool call, show original response
                            print(f"Assistant: {display_text}")
                            messages.append({"role": "assistant", "content": display_text})
                    else:
                        # Regular response, no tool call needed
                        print(f"Assistant: {display_text}")
                        messages.append({"role": "assistant", "content": display_text})
                
                print()  # Add spacing between conversations
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logging.error(f"Chat session error: {e}")

        # Cleanup
        print("🧹 Cleaning up servers...")
        await self.cleanup_servers()
              
async def main():
    config = Configuration()
    server_config = json.load(open("server_config.json"))
    servers = [Server(name, cfg) for name, cfg in server_config["mcpServers"].items()]
    llm_client = LLMClient(config.api_key)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()

if __name__ == "__main__":
    asyncio.run(main())
