"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Annotated, Literal
import uuid

import debugpy
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.messages import convert_to_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, data
from langgraph.runtime import Runtime
from pathlib import Path
from langgraph.graph.message import add_messages
from agent.helper_func import get_latest_relevant_memories, get_message_flatten_text_content, is_semantically_redundant
from agent.prompts import AGENT_PERSONA
from agent.tools import ALL_TOOLS
from agent.types import LLM
from langchain_core.messages import convert_to_messages
import uuid
from langchain_core.load import load

import os
from langchain_core.messages import SystemMessage
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from agent.models import ContextSchema, LLMConfiguration, MemoryExtraction



#Converters
def convert_to_valid_messages(meassges: list[AnyMessage] | list[dict]) -> list[AnyMessage]:
    """
    Converts serialized constructor dicts or standard dicts 
    into valid LangChain Message objects with guaranteed IDs.
    """
    result = []
    # 1. Handle the LangChain 'constructor' serialization format
    for data in meassges:
        message = None
        if isinstance(data, dict) and data.get("type") == "constructor":
            try:
                message = load(data)
            except Exception:
                kwargs = data.get("kwargs", {})
                message = convert_to_messages([kwargs])[0]
        else:
            message = convert_to_messages([data])[0]
        result.append(message)
    for msg in result:
        # 2. Ensure every message has a unique ID
        if not getattr(msg, 'id', None):
            try: 
                msg.id = str(uuid.uuid4())
            except Exception as e:
                print(f"Error occurred while generating ID for message: {e}")
    return result
#Reducers
def robust_message_reducer(left: list[AnyMessage], right: list[AnyMessage] | list[dict]) -> list[AnyMessage]:
    """Handles standard message updates AND serialized constructor dicts from Fork/Redo."""
    processed_right = []
    processed_left = convert_to_valid_messages(left)
    processed_right = convert_to_valid_messages(right)
    return add_messages(processed_left, processed_right)

# Classes - State, LLMConfiguration, ContextSchema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], robust_message_reducer]

# Get LLM
async def get_llm(llm_config: LLMConfiguration, tools: list = ALL_TOOLS) -> LLM:
    model = ChatGoogleGenerativeAI(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=None,
        timeout=None,
        max_retries=5,
        )
    llm = model.bind_tools(tools)
    return llm

#Node Helper Functions
def get_sanitized_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """
    Sanitizes history for Gemini without modifying the global state.
    - Merges consecutive HumanMessages.
    - Merges consecutive SystemMessages.
    - Removes 'Orphaned' AIMessages (tool calls with no response).
    - Ensures ToolMessages follow the specific AI turn that called them.
    """
    if not messages:
        return []

    processed = []

    pre_processed = []

    # 1. Identify the first non-system message
    first_msg_index = 0
    while first_msg_index < len(messages) and isinstance(messages[first_msg_index], SystemMessage):
        pre_processed.append(messages[first_msg_index])
        first_msg_index += 1
        
    # 2. Check if the NEXT message is an illegal Tool Call
    if first_msg_index < len(messages):
        target = messages[first_msg_index]
        if isinstance(target, AIMessage) and (target.tool_calls or target.additional_kwargs.get("function_call")):
            # INJECT a dummy Human Message to satisfy Gemini's protocol
            pre_processed.append(HumanMessage(content="Continuing previous task..."))
            print("Self-Healing: Injected 'Ghost' Human Message to fix summary truncation.")

    # 3. Add the rest of the messages
    pre_processed.extend(messages[first_msg_index:])
    
    for i, msg in enumerate(pre_processed):
        # 1. Handle consecutive SystemMessages (Merge)
        if isinstance(msg, SystemMessage) and processed and isinstance(processed[-1], SystemMessage):
            processed[-1] = SystemMessage(content=f"{processed[-1].content}\n\n{msg.content}")
            continue

        # 2. Handle consecutive HumanMessages (Merge)
        if isinstance(msg, HumanMessage) and processed and isinstance(processed[-1], HumanMessage):
            processed[-1] = HumanMessage(content=f"{processed[-1].content}\n\n{msg.content}")
            continue

        # 3. Handle AIMessages with Tool Calls
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Look ahead: is the next message a ToolMessage?
            # We check the original 'messages' list for this check
            has_tool_resp = (i + 1 < len(messages) and isinstance(messages[i + 1], ToolMessage))
            
            if not has_tool_resp:
                # This is an orphaned tool call. Gemini will 400.
                # We skip this message entirely to 'heal' the sequence.
                print(f"Self-Healing: Skipping orphaned tool call from AI (ID: {getattr(msg, 'id', 'unknown')})")
                continue

        # 4. Handle ToolMessages (Ensure they don't follow a HumanMessage)
        if isinstance(msg, ToolMessage) and processed and isinstance(processed[-1], HumanMessage):
            # This is a rare edge case if your image_processor injected a HumanMessage 
            # between a tool call and its response. We move the ToolMessage up.
            human_msg = processed.pop()
            processed.append(msg)
            processed.append(human_msg)
            continue

        processed.append(msg)

    return processed

def get_last_turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """
    Starts at the end and works backward to find the most recent HumanMessage,
    then returns that message and all subsequent messages (the 'turn').
    """
    if not messages:
        return []

    # Find the index of the last HumanMessage
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            # Return from that HumanMessage to the very end
            return messages[i:]
            
    # Fallback: if no human message is found, return everything (or empty)
    return messages

# Graph Nodes
async def summarizer(state: State, runtime: Runtime[ContextSchema]) -> State:
    llm_config = runtime.context.llm_configuration
    llm_with_tools = await get_llm(llm_config, tools=[]) # No tools for summarization step
    message_threshold = 15
    number_messages_to_keep = int(message_threshold*0.45)
    messages = state["messages"]
    cutoff_index = len(messages) - number_messages_to_keep

    # SAFETY: Move the cutoff back if we are in the middle of a Tool Call
    while cutoff_index > 0:
        # If the message at the cutoff is an AI Tool Call or a Tool Message, 
        # move the cutoff back so we don't break the chain.
        if isinstance(messages[cutoff_index], (ToolMessage, AIMessage)):
            cutoff_index -= 1
        else:
            break

    result = None
    if len(messages) < message_threshold:
        result = State(messages=[])
    if len(messages) >= message_threshold:
        
        system_prompt = SystemMessage(content="You are a helpful assistant that summarizes conversations, preserving all file paths mentioned.")
        summary_prompt = HumanMessage(content="""
                    Summarize the previous conversation and return a concise summary that captures all important details, especially any file paths or tool outputs. 
                    Be sure to retain any information that might be relevant for future context. 
                    The summary should be brief but comprehensive.
                    The summary should be 10 sentences long maximum.
                    The summary should have the following format:
                    The following content is a summary of the conversation prior: <insert summary here>
                                      """)
        past_messages = messages[:cutoff_index] 
        llm_input = past_messages + [system_prompt] + [summary_prompt]
        ai_response = await llm_with_tools.ainvoke(llm_input)
        ai_response_as_syastem_message = SystemMessage(content=ai_response.content[0]["text"])
        ai_response_as_syastem_message.id = str(uuid.uuid4())
        removed_past_messages = [RemoveMessage(id=msg.id) for msg in messages[:cutoff_index]]
        removed_messages_to_keep = [RemoveMessage(id=msg.id) for msg in messages[cutoff_index:]]
        messages_to_keep_with_new_id = []
        for msg in messages[cutoff_index:]:
            new_msg = msg.model_copy()
            new_msg.id = str(uuid.uuid4())
            messages_to_keep_with_new_id.append(new_msg)
        messages = [ai_response_as_syastem_message] + removed_past_messages + removed_messages_to_keep + messages_to_keep_with_new_id
        result = State(messages=messages)
    return result

async def brain(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore) -> State:
    # 1. Setup and context gathering
    llm_config = runtime.context.llm_configuration
    user_messages = get_sanitized_messages(state["messages"])
    namespace = ("memories", runtime.context.user_id)
    
    memory_context = ""

    # 2. Semantic Memory Retrieval (Only on new Human turns)
    if not isinstance(user_messages[-1], ToolMessage):
        query_text = next((str(m.content) for m in reversed(user_messages) if isinstance(m, HumanMessage)), "")
        if query_text:
            memory_context = await get_latest_relevant_memories(query_text, namespace, store)

    # 3. Message Construction
    system_content = f"{runtime.context.persona}\n{memory_context}"
    final_messages = [SystemMessage(content=system_content)] + user_messages

    # 4. LLM Execution
    llm_with_tools = await get_llm(llm_config)
    ai_message = await llm_with_tools.ainvoke(final_messages)
    
    # Flatten multi-block content for LangSmith/State consistency
    ai_message = get_message_flatten_text_content(ai_message)
    
    return State(messages=[ai_message])

async def image_processor(state: State, runtime: Runtime[ContextSchema]) -> State:
    """
    Scans message history for the VISUAL_INJECTION_64 signal 
    and converts text-based Base64 tool outputs into vision-capable blocks.
    """
    messages = state["messages"]
    refined_messages = []
    
    for msg in messages:
        # Check if the tool output contains the magic signal
        if isinstance(msg, ToolMessage) and "VISUAL_INJECTION_64:" in msg.content and "VISUAL_INJECTION_64: <your_base64_string>" not in msg.content:
            try:
                # Extract the Base64 payload
                parts = msg.content.split("VISUAL_INJECTION_64:")
                b64_data = parts[1].strip()
                
                # Create the formatted vision message
                vision_msg = HumanMessage(
                    content=[
                        {"type": "text", "text": "REPL Image Processing Complete. Visual context attached below:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
                    ]
                )
                refined_messages.append(vision_msg)
            except Exception as e:
                # Fallback if the string was malformed
                refined_messages.append(HumanMessage(content=f"Error decoding visual injection: {str(e)}"))
                refined_messages.append(msg)

    result = State(messages=refined_messages)   
    return result

async def memory_saver(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore):
    """
    Orchestrates the extraction and deduplicated saving of user memories.
    """
    user_id = runtime.context.user_id
    namespace = ("memories", user_id)
    thread_id = runtime.execution_info.thread_id

    # 1. Extract potential insights from the last exchange
    last_messages = get_last_turn_messages(state["messages"])
    
    llm = await get_llm(runtime.context.llm_configuration, [])
    model = llm.with_structured_output(MemoryExtraction)
    
    system_prompt = """
    You are a memory-distillation assistant. Extract NEW facts or preferences about the user.
    - CATEGORY: Use snake_case (e.g., 'food_preference').
    - TYPE: 'fact' or 'user_preference'.
    - CONTENT: A clear, standalone statement.
    """
    
    extracted = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"CONVERSATION TO REVIEW:\n{last_messages}")
    ])
    
    new_insights = extracted.insights

    # 2. Process and save unique insights
    for insight in new_insights:
        # Targeted check: Does this exact idea already exist in our Vector Store?
        if await is_semantically_redundant(insight.content, namespace, store):
            continue # Skip saving to prevent the image_cd47a0.png duplication issue

        # 3. Save the fresh insight with full metadata
        memory_id = str(uuid.uuid4())
        value = {
            "content": insight.content,
            "type": insight.type,
            "category": insight.category.lower(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "thread_id": thread_id
        }
        await store.aput(namespace, memory_id, value)

    return State(messages=[])

# Conditional Edges
def decide_start(state: State, runtime: Runtime[ContextSchema]) -> Literal["summarizer", "brain_node"]:
    message_threshold = runtime.context.message_threshold

    messages = state["messages"]

    result = None
    if len(messages) < message_threshold:
        result = "brain_node"
    if len(messages) >= message_threshold:
        result = "summarizer"
    return result

def decide_image_processing(state: State, runtime: Runtime[ContextSchema]) -> Literal["image_processor", "brain_node"]:
    messages = state["messages"]
    image_processing_required = False;
    
    for msg in messages:
        # Check if the tool output contains the magic signal
        if isinstance(msg, ToolMessage) and "VISUAL_INJECTION_64:" in msg.content:
            image_processing_required = True
            break
    result = None
    if image_processing_required:
        result = "image_processor"
    else:
        result = "brain_node"
    return result
def decide_after_brain(state: State):
    route = tools_condition(state)
    if route == END:
        return "memory_saver"
    return "tools"
# Graph Definition
def get_graph():
    try:
        # 0.0.0.0 is required inside Docker!
        debugpy.listen(("0.0.0.0", 5679))
    except Exception:
        pass # Prevents the worker crash we saw earlier

    workflow = StateGraph(State, context_schema=ContextSchema)
    
    # Nodes
    workflow.add_node("brain_node", brain)
    workflow.add_node("summarizer", summarizer)
    workflow.add_node("image_processor", image_processor)
    workflow.add_node("memory_saver", memory_saver)
    workflow.add_node("tools", ToolNode(ALL_TOOLS)) # 'tools' is your list of @tool functions
    
    # Edges
    workflow.add_conditional_edges(
        START,
        decide_start,
        {
            "summarizer": "summarizer",
            "brain_node": "brain_node"
        }
    )
    workflow.add_edge("summarizer", "brain_node")
    workflow.add_conditional_edges(
        "brain_node",
        # This helper function automatically checks if the LLM called a tool
        decide_after_brain, 
        {
            "tools": "tools", # If tool called, go to 'tools' node
            "memory_saver": "memory_saver"          # If no tool called, go to memory node
        }
    )
    workflow.add_conditional_edges(
        "tools",
        decide_image_processing,
        {
            "image_processor": "image_processor",
            "brain_node": "brain_node"
        } 
    )
    workflow.add_edge("image_processor", "brain_node")  # After image processing, go back to brain
    workflow.add_edge("memory_saver", END)
    return workflow

# Embedding Function
# 2. Add this wrapper function specifically for LangGraph API
async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """LangGraph worker will call this async function to embed text."""
    return await embedding_object.aembed_documents(texts)

# Compiled Graph Definition
def get_compiled_graph() -> StateGraph:
    graphName = "Agent"
    result = (
        get_graph()
            .compile(
                name=graphName,
            )
    )
    return result;

#LOAD Variables - embedding object, graph
load_dotenv()
embedding_object = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    output_dimensionality=768 
)
graph = get_compiled_graph()
# tools CRUD + Execute
#   create files on the file system.
#   read files on the file system.
#   update files on the file system.
#   delete files on the file system.
#   execute bash scripts on the system
#   create new tools and add it to the agent. (how do I do this?
# We are reaching into self modifying code territory.



