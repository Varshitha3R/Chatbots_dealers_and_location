import sqlite3
import os
import json
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from flask_cors import CORS
import openai
import requests
from bs4 import BeautifulSoup
import uuid
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.prebuilt import ToolNode


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize SQLite database
def initialize_db():
    connection = sqlite3.connect("chatbot_memory.db")
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Conversations (
            session_id TEXT PRIMARY KEY,
            messages TEXT
        )
    """)
    connection.commit()
    connection.close()

# Call this function to initialize the database at the start
initialize_db()

def save_memory(session_id, messages):
    connection = sqlite3.connect("chatbot_memory.db")
    cursor = connection.cursor()
    messages_json = json.dumps(messages)
    cursor.execute("""
        INSERT OR REPLACE INTO Conversations (session_id, messages)
        VALUES (?, ?)
    """, (session_id, messages_json))
    connection.commit()
    connection.close()

def load_memory(session_id):
    connection = sqlite3.connect("chatbot_memory.db")
    cursor = connection.cursor()
    cursor.execute("SELECT messages FROM Conversations WHERE session_id = ?", (session_id,))
    result = cursor.fetchone()
    connection.close()
    if result:
        return json.loads(result[0])
    return []

# Initialize the OpenAI client
client = openai.OpenAI()

# Function to fetch dealer location using OpenAI API
def FetchDealerLocation(question):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Extract a ZIP code, address, city, or state from the following question. If none exists, respond with 'I don't know'."},
            {"role": "user", "content": f"Question: {question}"}
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Function to scrape dealer information for a given location
def ScrapeDealerInfo(location):
    url = f"https://www.msisurfaces.com/dealer-locator/countertops-flooring-hardscaping-stile/{location}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        dealers_info = []

        for h4_tag in soup.find_all('h4'):
            dealer_data = {}
            b_tag = h4_tag.find('b')
            if b_tag:
                dealer_data['Dealer Name'] = b_tag.get_text(strip=True)

                div_tag = h4_tag.find_next('div')
                if div_tag:
                    span_tags = div_tag.find_all('span')
                    if span_tags:
                        location = span_tags[0].get_text(strip=True)
                        products = [span.get_text(strip=True) for span in span_tags[1:]]
                        dealer_data['Location'] = location
                        dealer_data['Products'] = products

                dealers_info.append(dealer_data)
        return dealers_info
    else:
        return []

# Function to extract dealer information based on a user's question
def ExtractDealerInfo(question):
    extracted_location = FetchDealerLocation(question)

    if extracted_location and extracted_location.lower() != "i don't know":
        dealers_info = ScrapeDealerInfo(extracted_location)

        if dealers_info:
            context = f"Dealer information for {extracted_location}: {dealers_info}"

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"""You are an MSI AI assistant. Use the context below to answer questions about dealers.
                    Provide answers in HTML format using <li>, <ol>, <p>, <b>, and <i> tags only.
                    Context: {context}"""},
                    {"role": "user", "content": f"UserInput: {question}"}
                ],
                temperature=0,
            )

            return response.choices[0].message.content.strip()
        else:
            return "No dealer information found for the given location."
    else:
        return "No valid ZIP code or city found in the query."

# Define LangChain prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant for MSI specializing in surfaces. Assist users with queries about dealers and products."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Define LangGraph state
class State(TypedDict):
    messages: Annotated[List[dict], "The messages in the conversation"]
    session_id: Annotated[str, "The session ID for the conversation"]
    step_count: Annotated[int, "The number of steps taken in the conversation"]

# Define LangGraph nodes
def human(state: State) -> State:
    return state

def ai(state: State) -> State:
    messages = state["messages"]
    session_id = state["session_id"]

    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    response = llm(messages)

    messages.append({"role": "ai", "content": response.content})
    save_memory(session_id, messages)

    state["messages"] = messages
    state["step_count"] += 1
    return state

def dealer_info(state: State) -> State:
    messages = state["messages"]
    last_human_message = next(msg["content"] for msg in reversed(messages) if msg["role"] == "human")

    if "dealer" in last_human_message.lower() or "location" in last_human_message.lower():
        dealer_info = ExtractDealerInfo(last_human_message)
        messages.append({"role": "ai", "content": f"Here's some additional dealer information: {dealer_info}"})

    state["messages"] = messages
    state["step_count"] += 1
    return state

# Define routing logic
def should_continue(state: State):
    if state["step_count"] >= 5:  # Limit to 5 steps to avoid recursion error
        return END
    return "human"

# Build LangGraph
workflow = StateGraph(State)

workflow.add_node("human", human)
workflow.add_node("ai", ai)
workflow.add_node("dealer_info", dealer_info)

workflow.set_entry_point("human")
workflow.add_edge("human", "ai")
workflow.add_edge("ai", "dealer_info")
workflow.add_conditional_edges("dealer_info", should_continue)

graph = workflow.compile()

@app.route('/var', methods=['POST', 'GET'])
def chat_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        user_question = data.get('question')
        session_id = data.get('session_id', str(uuid.uuid4()))
    elif request.method == 'GET':
        user_question = request.args.get('question')
        session_id = request.args.get('session_id', str(uuid.uuid4()))
    else:
        return jsonify({"error": "Invalid request method"}), 405

    if not user_question:
        return jsonify({"error": "Question is missing"}), 400

    chat_history = load_memory(session_id)

    if "previous conversation" in user_question.lower():
        if chat_history:
            return jsonify({"response": "Here's the previous conversation:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history]), "session_id": session_id})
        else:
            return jsonify({"response": "I'm sorry, but there's no previous conversation stored for this session.", "session_id": session_id})

    # Add user's new question to the conversation history
    chat_history.append({"role": "human", "content": user_question})

    # Use LangGraph to process the conversation
    state = {"messages": chat_history, "session_id": session_id, "step_count": 0}
    result = graph.invoke(state)

    # Extract the AI's response
    ai_response = next(msg["content"] for msg in reversed(result["messages"]) if msg["role"] == "ai")

    return jsonify({"response": ai_response, "session_id": session_id})

if __name__ == "__main__":
    initialize_db()
    app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)
