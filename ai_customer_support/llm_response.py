import requests
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

def generate_ollama_response(user_query, context):
    prompt = f"""
You are a helpful and professional customer support assistant.

Guidelines:
- Use concise and polite language
- If context is unclear, ask for clarification instead of guessing
- Use bullet points for long answers

Context:
{context}

User Question:
{user_query}

Response:
"""
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post("https://api.ollama.com/v1/generate", headers=headers, json=payload)
        data = response.json()
        answer = data.get("completion", "")
        
        # Simple confidence heuristic: if answer too short or generic, fallback to human
        if len(answer) < 20 or "I'm not sure" in answer:
            return "⚠️ Escalate to human agent"
        
        return answer
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"
