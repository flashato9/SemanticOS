from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from agent.prompts import AGENT_PERSONA

class LLMConfiguration(BaseModel):
    "the configuration for the llm"
    model_name: str = "gemini-2.5-flash"
    temperature: float = 1.0
    
class ToolsConfig(BaseModel):
    "the configuration for the tools"
    workspace_root: str = Field(
        default="./static/agent_files",
        description="The root directory for all file operations. Agents cannot access files outside this directory."
    )
    max_bytes: int = Field(
        default=50000,
        description="The maximum number of bytes to read from a file. If a file exceeds this, only the first max_bytes will be returned."
    )  
    current_relative_path: str = Field(
        default=".",
        description="The agent's current working directory relative to the workspace root. This allows the agent to 'navigate' within the sandbox."
    )
    execution_timeout : int = Field(
        default=30,
        description="The maximum time in seconds to allow a script to run when using the execute_file tool. This prevents infinite loops or long-running processes."
    )
    
class ContextSchema(BaseModel):
    llm_configuration: LLMConfiguration = LLMConfiguration()
    user_id: str = "default_user"
    persona: str = Field(
        default = (
           AGENT_PERSONA
           ),
        description = "The persona for the assistant",
        json_schema_extra={
            "langgraph_nodes": ["brain_node"],
            "langgraph_type": "prompt"
        }
    )
    message_threshold: int = Field(
        default=100,
        description="The number of recent messages to include in the context for decision making. Older messages will be summarized into the 'meaning' field of ToolResult."
    )
    tool_config: ToolsConfig = ToolsConfig()
    
class ToolResult(BaseModel):
    """The universal response object for all agentic tools."""
    
    success: bool = Field(description="Whether the operation completed without errors")
    output: Any = Field(description="The primary data produced by the tool (e.g., file path, API JSON)")
    error: Optional[str] = Field(None, description="Detailed error message if success is False")
    meaning: str = Field(description="A human-readable summary for the agent's context")
    

    def to_state_update(self) -> dict:
        """Helper to format the result for LangGraph state updates."""
        return {
            "artifact_log": [self.model_dump()],
            "messages": [self.meaning]
        }
        
# Define a schema for the memory we want to extract
class MemoryExtraction(BaseModel):
    facts: list[str] = Field(description="New facts about the user or project")
    style_preferences: list[str] = Field(description="Preferences for how the AI should behave or code")
    
class MemoryInsight(BaseModel):
    content: str = Field(description="The actual fact or preference discovered.")
    type: Literal["fact", "user_preference"] = Field(description="The classification of the info.")
    category: str = Field(description="The grouping key (e.g., 'food_pref', 'coding_style'). Use snake_case.")

class MemoryExtraction(BaseModel):
    insights: List[MemoryInsight]   