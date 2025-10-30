from flask import Flask, render_template, request, jsonify
from rag_pipeline import get_context
from llm_response import generate_ollama_response
from conversation_memory import add_to_memory, get_recent_context
import uuid

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_query = data.get("query", "")
    session_id = data.get("session_id", str(uuid.uuid4()))
    
    # Retrieve context from Qdrant
    context = get_context(user_query)
    
    # Add recent conversation context
    recent = get_recent_context(session_id)
    full_context = context + "\n" + recent
    
    # Generate AI response
    ai_response = generate_ollama_response(user_query, full_context)
    
    # Store in memory
    add_to_memory(session_id, user_query, ai_response)
    
    return jsonify({"response": ai_response, "session_id": session_id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
