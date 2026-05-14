import copy

from langchain_core.messages import AIMessage
from langgraph.store.base import BaseStore

def get_message_flatten_text_content(message: AIMessage) -> AIMessage:
    """
    Standardizes AIMessage content for LangSmith readability.
    Joins multiple text blocks into one, while preserving tool calls, 
    images, or other non-text blocks.
    """
    if isinstance(message.content, str):
        return message

    if isinstance(message.content, list):
        new_content = []
        text_parts = []
        
        for block in message.content:
            # 1. Collect text blocks for merging
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            
            # 2. Keep other blocks (tool_use, image, etc.) as they are
            elif isinstance(block, dict):
                # If we have accumulated text, flush it before adding a non-text block
                if text_parts:
                    new_content.append({"type": "text", "text": "".join(text_parts)})
                    text_parts = []
                new_content.append(block)
            
            # 3. Handle raw strings mixed in lists
            elif isinstance(block, str):
                text_parts.append(block)

        # Final flush of accumulated text
        if text_parts:
            merged_text = "".join(text_parts)
            # If the ONLY thing in the message was text, LangChain prefers a string
            if not new_content:
                new_content = merged_text
            else:
                new_content.append({"type": "text", "text": merged_text})

        # Create the new message preserving all metadata/tool_calls
        new_message = copy.deepcopy(message)
        new_message.content = new_content
        return new_message

    return message

async def get_latest_relevant_memories(
    query: str, 
    namespace: tuple, 
    store: BaseStore, 
    candidate_limit: int = 15, 
    final_limit: int = 5
) -> str:
    """
    Finds relevant memories and ensures only the most recent version 
    of any given category is returned.
    """
    search_results = await store.asearch(namespace, query=query, limit=candidate_limit)
    if not search_results:
        return ""
    default_created_at = "1970-01-01T00:00:00Z"
    # Sort candidates by created_at descending (newest first)
    sorted_results = sorted(
        search_results, 
        key=lambda x: x.value.get("created_at", default_created_at), 
        reverse=True
    )

    resolved_memories = {}
    for res in sorted_results:
        category = res.value.get("category", "uncategorized")
        # First one found for a category is the latest; lock it in
        if category not in resolved_memories:
            resolved_memories[category] = res.value["content"]

    # Limit to the top unique results and format
    final_facts = list(resolved_memories.values())[:final_limit]
    return "\nRelevant Long-term Preferences:\n" + "\n".join([f"- {f}" for f in final_facts])

async def is_semantically_redundant(insight_content: str, namespace: tuple, store: BaseStore, threshold: float = 0.9) -> bool:
    """
    Checks if a similar insight already exists in the store to prevent 'bagel duplication'.
    """
    existing_matches = await store.asearch(
        namespace,
        query=insight_content,
        limit=1
    )

    if existing_matches:
        top_match = existing_matches[0]
        # If the vector similarity is higher than our threshold, it's a duplicate
        if top_match.score > threshold:
            return True
            
    return False
