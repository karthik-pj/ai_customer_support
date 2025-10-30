# Simple in-memory conversation memory per session
conversation_history = {}

def add_to_memory(session_id, user_query, ai_response):
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append({"user": user_query, "ai": ai_response})

def get_recent_context(session_id, last_n=3):
    if session_id not in conversation_history:
        return ""
    recent = conversation_history[session_id][-last_n:]
    context_text = " ".join([f"User: {m['user']} AI: {m['ai']}" for m in recent])
    return context_text
