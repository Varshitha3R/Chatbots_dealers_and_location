
from IPython.display import Image, display
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
import requests
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import csv
import os
import uuid
import numpy as np
import openai
from flask import Flask, request, jsonify
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from bs4 import BeautifulSoup
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.schema import SystemMessage
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

MEMORY_DIR = "chatbot_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def save_memory(session_id, messages):
    file_path = os.path.join(MEMORY_DIR, f"{session_id}.json")
    with open(file_path, 'w') as f:
        json.dump(messages, f)

def load_memory(session_id):
    file_path = os.path.join(MEMORY_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

class State(TypedDict):
    messages: Annotated[list, add_messages]

def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result."""
    return a * b

def add(a: int, b: int) -> int:
    """Adds two integers and returns the result."""
    return a + b

def FetchDealerLocation(question):
    """Extract zipcode or address or location or city or state from the question"""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Please extract zipcode or address or location or city or state from the following question and if you don't find it tell i don't know"},
            {"role": "user", "content": f"Question: {question}"}
        ],
        temperature=0
    )
    address = response.choices[0].message.content
    print(address)
    return address

def ScrapeDealerInfo(location):
    """Scrape dealer information from a location"""
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
                    else:
                        dealer_data['Details'] = "No additional details found."
                else:
                    dealer_data['Details'] = "No div section after h4."

                dealers_info.append(dealer_data)
        return dealers_info
    else:
        return []

def ExtractDealerInfo(question):
    """Extract dealer information from a question"""
    extracted_location = FetchDealerLocation(question)

    if extracted_location:
        dealers_info = ScrapeDealerInfo(extracted_location)

        if dealers_info:
            context = f"Dealer information for {extracted_location}: {dealers_info}"

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"""You are an MSI AI assistant with the ability to analyze the context and provide answers based on the dealer information scraped from the provided ZIP code.
                    Answer the question clearly and organize the information in bullet points for easy readability.
                    Keep the responses in HTML format - use only these tags: <li>, <ol>, <p>, <b>, <i>.
                    Context: {context}"""},
                    {"role": "user", "content": f"UserInput: {question}"}
                ],
                temperature=0
            )

            result = response.choices[0].message.content
            return result
        else:
            return "No dealer information found for the given location."
    else:
        return "No valid ZIP code or city found in the query."

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant for MSI, a company specializing in surfaces. Your role is to assist users with queries about dealers, products, and basic arithmetic operations. Use the available tools when necessary to provide accurate information."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

tools = [ScrapeDealerInfo, ExtractDealerInfo, multiply, add]
llm_with_tools = llm.bind_tools(tools)
bound = prompt | llm_with_tools

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

graph_builder = StateGraph(State)

# define nodes
graph_builder.add_node("assistant", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

# define edges
graph_builder.add_edge(START, "assistant")
graph_builder.add_conditional_edges("assistant", tools_condition)
graph_builder.add_edge("tools", "assistant")
react_graph = graph_builder.compile(checkpointer=memory)

@app.route('/chat', methods=['POST', 'GET'])
def chat_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        user_question = data.get('question')
        session_id = data.get('session_id', 'default')
    elif request.method == 'GET':
        user_question = request.args.get('question')
        session_id = request.args.get('session_id', 'default')
    else:
        return jsonify({"error": "Invalid request method"}), 405

    if not user_question:
        return jsonify({"error": "Question is missing"}), 400

    # Load the conversation history
    chat_history = load_memory(session_id)

    # Convert the loaded history to Message objects
    messages = [
        SystemMessage(content="You are an AI assistant for MSI, a company specializing in surfaces. Your role is to assist users with queries about dealers, products, and basic information. Remember to maintain context from previous messages in the conversation.")
    ]
    for msg in chat_history:
        if msg['role'] == 'human':
            messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'ai':
            messages.append(AIMessage(content=msg['content']))

    # Add the new user message
    messages.append(HumanMessage(content=user_question))

    # Get AI's response
    response = llm(messages)

    # Check if dealer information is requested
    if "dealer" in user_question.lower() or "location" in user_question.lower():
        dealer_info = ExtractDealerInfo(user_question)
        response.content += f"\n\nHere's some additional dealer information: {dealer_info}"

    # Add AI's response to the chat history
    chat_history.append({"role": "human", "content": user_question})
    chat_history.append({"role": "ai", "content": response.content})

    # Save the updated chat history
    save_memory(session_id, chat_history)

    return jsonify({"response": response.content, "session_id": session_id})

if __name__ == "__main__":
    app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)
# # code with memory



# from IPython.display import Image, display
# from langgraph.graph import StateGraph, START
# from langchain_openai import ChatOpenAI
# import requests
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langgraph.graph import MessagesState
# from langgraph.prebuilt import ToolNode, tools_condition
# from typing import Annotated
# from typing_extensions import TypedDict
# from langgraph.graph.message import add_messages
# import csv
# import os
# import uuid
# import numpy as np
# import openai
# from flask import Flask, request, jsonify
# from langchain.embeddings import OpenAIEmbeddings
# from dotenv import load_dotenv
# from flask_cors import CORS
# from sklearn.metrics.pairwise import cosine_similarity
# from typing import List, Dict
# from bs4 import BeautifulSoup
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
# from langchain.llms import OpenAI
# from langchain.schema import SystemMessage
# import json
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# app = Flask(__name__)
# CORS(app)

# # Load environment variables
# load_dotenv()

# # Set up OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")
# if not openai.api_key:
#     raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# MEMORY_DIR = "chatbot_memory"
# os.makedirs(MEMORY_DIR, exist_ok=True)

# def save_memory(session_id, messages):
#     file_path = os.path.join(MEMORY_DIR, f"{session_id}.json")
#     with open(file_path, 'w') as f:
#         json.dump(messages, f)

# def load_memory(session_id):
#     file_path = os.path.join(MEMORY_DIR, f"{session_id}.json")
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             return json.load(f)
#     return []

# class State(TypedDict):
#     messages: Annotated[list, add_messages]

# # Function to process CSV data
# def multiply(a: int, b: int) -> int:
#     """Multiplies two integers and returns the result."""
#     return a * b

# def add(a: int, b: int) -> int:
#     """Adds two integers and returns the result."""
#     return a + b

# # Function to fetch dealer location from a query
# def FetchDealerLocation(question):
#     """Please extract zipcode or address or location or city or state from the following question and if you don't find it tell i don't know"""
#     system_role = "Please extract zipcode or address or location or city or state from the following question and if you don't find it tell i don't know\n\n Question:" + question + "?"

#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "assistant", "content": system_role},
#             {"role": "user", "content": "UserInput: " + question}
#         ],
#         temperature=0
#     )

#     address = response.choices[0].message.content
#     print(address)
#     return address

# # Function to scrape dealer information from a location
# def ScrapeDealerInfo(location):
#     """Scrape dealer information from a location"""
#     url = f"https://www.msisurfaces.com/dealer-locator/countertops-flooring-hardscaping-stile/{location}"
#     response = requests.get(url)

#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, 'html.parser')
#         dealers_info = []

#         for h4_tag in soup.find_all('h4'):
#             dealer_data = {}
#             b_tag = h4_tag.find('b')
#             if b_tag:
#                 dealer_data['Dealer Name'] = b_tag.get_text(strip=True)

#                 div_tag = h4_tag.find_next('div')
#                 if div_tag:
#                     span_tags = div_tag.find_all('span')
#                     if span_tags:
#                         location = span_tags[0].get_text(strip=True)
#                         products = [span.get_text(strip=True) for span in span_tags[1:]]
#                         dealer_data['Location'] = location
#                         dealer_data['Products'] = products
#                     else:
#                         dealer_data['Details'] = "No additional details found."
#                 else:
#                     dealer_data['Details'] = "No div section after h4."

#                 dealers_info.append(dealer_data)
#         return dealers_info
#     else:
#         return []

# # Function to extract dealer information from a question
# def ExtractDealerInfo(question):
#     """Extract dealer information from a question"""
#     extracted_location = FetchDealerLocation(question)

#     if extracted_location:
#         dealers_info = ScrapeDealerInfo(extracted_location)

#         if dealers_info:
#             context = f"Dealer information for {extracted_location}: {dealers_info}"

#             system_role = f"""You are an MSI AI assistant with the ability to analyze the context and provide answers based on the dealer information scraped from the provided ZIP code.
#             Answer the question clearly and organize the information in bullet points for easy readability.
#             Keep the responses in HTML format - use only these tags: <li>, <ol>, <p>, <b>, <i>.
#             Context: {context}"""

#             response = openai.ChatCompletion.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "assistant", "content": system_role},
#                     {"role": "user", "content": "UserInput: " + question}
#                 ],
#                 temperature=0
#             )

#             result = response.choices[0].message.content
#             return result
#         else:
#             return "No dealer information found for the given location."
#     else:
#         return "No valid ZIP code or city found in the query."

# # # Set up session data directory
# # SESSION_DATA_DIR = "session_data"
# # os.makedirs(SESSION_DATA_DIR, exist_ok=True)

# # def getSessionMemory(sessionID):
# #     """Retrieve or initialize session-specific memory."""
# #     file_path = os.path.join(SESSION_DATA_DIR, f"{sessionID}.json")
# #     if os.path.exists(file_path):
# #         with open(file_path, 'r') as f:
# #             data = json.load(f)
# #         memory = ConversationBufferMemory(k=10, memory_key="chat_history", return_messages=True)
# #         for message in data['messages']:
# #             if message['type'] == 'human':
# #                 memory.chat_memory.add_user_message(message['content'])
# #             elif message['type'] == 'ai':
# #                 memory.chat_memory.add_ai_message(message['content'])
# #     else:
# #         memory = ConversationBufferMemory(k=10, memory_key="chat_history", return_messages=True)
# #     return memory

# # def saveSessionMemory(sessionID, memory):
# #     file_path = os.path.join(SESSION_DATA_DIR, f"{sessionID}.json")
# #     data = {
# #         'messages': [
# #             {'type': 'human' if isinstance(m, HumanMessage) else 'ai', 'content': m.content}
# #             for m in memory.chat_memory.messages
# #         ]
# #     }
# #     with open(file_path, 'w') as f:
# #         json.dump(data, f)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an AI assistant for MSI, a company specializing in surfaces. Your role is to assist users with queries about dealers, products, and basic arithmetic operations. Use the available tools when necessary to provide accurate information."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])

# llm = ChatOpenAI(temperature=0, api_key="", model_name="gpt-4-0125-preview")

# tools = [ScrapeDealerInfo, ExtractDealerInfo, multiply, add]
# llm_with_tools = llm.bind_tools(tools)
# bound = prompt | llm_with_tools

# def chatbot(state: State):
#     return {"messages": [llm_with_tools.invoke(state["messages"])]}

# from langgraph.checkpoint.memory import MemorySaver
# memory = MemorySaver()

# graph_builder = StateGraph(State)

# # define nodes
# graph_builder.add_node("assistant", chatbot)
# graph_builder.add_node("tools", ToolNode(tools))

# # define edges
# graph_builder.add_edge(START, "assistant")
# graph_builder.add_conditional_edges("assistant", tools_condition)
# graph_builder.add_edge("tools", "assistant")
# react_graph = graph_builder.compile(checkpointer=memory)

# @app.route('/chat', methods=['POST', 'GET'])
# def chat_endpoint():
#     if request.method == 'POST':
#         data = request.get_json()
#         user_question = data.get('question')
#         session_id = data.get('session_id', 'default')
#     elif request.method == 'GET':
#         user_question = request.args.get('question')
#         session_id = request.args.get('session_id', 'default')
#     else:
#         return jsonify({"error": "Invalid request method"}), 405

#     if not user_question:
#         return jsonify({"error": "Question is missing"}), 400

#     # Load the conversation history
#     chat_history = load_memory(session_id)

#     # Convert the loaded history to Message objects
#     messages = [
#         SystemMessage(content="You are an AI assistant for MSI, a company specializing in surfaces. Your role is to assist users with queries about dealers, products, and basic information. Remember to maintain context from previous messages in the conversation.")
#     ]
#     for msg in chat_history:
#         if msg['role'] == 'human':
#             messages.append(HumanMessage(content=msg['content']))
#         elif msg['role'] == 'ai':
#             messages.append(AIMessage(content=msg['content']))

#     # Add the new user message
#     messages.append(HumanMessage(content=user_question))

#     # Get AI's response
#     response = llm(messages)

#     # Check if dealer information is requested
#     if "dealer" in user_question.lower() or "location" in user_question.lower():
#         dealer_info = ExtractDealerInfo(user_question)
#         response.content += f"\n\nHere's some additional dealer information: {dealer_info}"

#     # Add AI's response to the chat history
#     chat_history.append({"role": "human", "content": user_question})
#     chat_history.append({"role": "ai", "content": response.content})

#     # Save the updated chat history
#     save_memory(session_id, chat_history)

#     return jsonify({"response": response.content, "session_id": session_id})

# if __name__ == "__main__":
#     app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)

# session memeory no functions



# from flask import Flask, request, jsonify
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from dotenv import load_dotenv
# from flask_cors import CORS
# import os
# import json
# import openai
# import requests
# from bs4 import BeautifulSoup

# app = Flask(__name__)
# CORS(app)

# # Load environment variables
# load_dotenv()

# # Set up OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")
# if not openai.api_key:
#     raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# # Set up memory directory
# MEMORY_DIR = "chatbot_memory"
# os.makedirs(MEMORY_DIR, exist_ok=True)

# def save_memory(session_id, messages):
#     file_path = os.path.join(MEMORY_DIR, f"{session_id}.json")
#     with open(file_path, 'w') as f:
#         json.dump(messages, f)

# def load_memory(session_id):
#     file_path = os.path.join(MEMORY_DIR, f"{session_id}.json")
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             return json.load(f)
#     return []

# def multiply(a: int, b: int) -> int:
#     """Multiplies two integers and returns the result."""
#     return a * b

# def add(a: int, b: int) -> int:
#     """Adds two integers and returns the result."""
#     return a + b

# def FetchDealerLocation(question):
#     """Please extract zipcode or address or location or city or state from the following question and if you don't find it tell i don't know"""
#     system_role = "Please extract zipcode or address or location or city or state from the following question and if you don't find it tell i don't know\n\n Question:" + question + "?"

#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "assistant", "content": system_role},
#             {"role": "user", "content": "UserInput: " + question}
#         ],
#         temperature=0
#     )

#     address = response.choices[0].message.content
#     print(address)
#     return address

# def ScrapeDealerInfo(location):
#     """Scrape dealer information from a location"""
#     url = f"https://www.msisurfaces.com/dealer-locator/countertops-flooring-hardscaping-stile/{location}"
#     response = requests.get(url)

#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, 'html.parser')
#         dealers_info = []

#         for h4_tag in soup.find_all('h4'):
#             dealer_data = {}
#             b_tag = h4_tag.find('b')
#             if b_tag:
#                 dealer_data['Dealer Name'] = b_tag.get_text(strip=True)

#                 div_tag = h4_tag.find_next('div')
#                 if div_tag:
#                     span_tags = div_tag.find_all('span')
#                     if span_tags:
#                         location = span_tags[0].get_text(strip=True)
#                         products = [span.get_text(strip=True) for span in span_tags[1:]]
#                         dealer_data['Location'] = location
#                         dealer_data['Products'] = products
#                     else:
#                         dealer_data['Details'] = "No additional details found."
#                 else:
#                     dealer_data['Details'] = "No div section after h4."

#                 dealers_info.append(dealer_data)
#         return dealers_info
#     else:
#         return []

# def ExtractDealerInfo(question):
#     """Extract dealer information from a question"""
#     extracted_location = FetchDealerLocation(question)

#     if extracted_location:
#         dealers_info = ScrapeDealerInfo(extracted_location)

#         if dealers_info:
#             context = f"Dealer information for {extracted_location}: {dealers_info}"

#             system_role = f"""You are an MSI AI assistant with the ability to analyze the context and provide answers based on the dealer information scraped from the provided ZIP code.
#             Answer the question clearly and organize the information in bullet points for easy readability.
#             Keep the responses in HTML format - use only these tags: <li>, <ol>, <p>, <b>, <i>.
#             Context: {context}"""

#             response = openai.ChatCompletion.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "assistant", "content": system_role},
#                     {"role": "user", "content": "UserInput: " + question}
#                 ],
#                 temperature=0
#             )

#             result = response.choices[0].message.content
#             return result
#         else:
#             return "No dealer information found for the given location."
#     else:
#         return "No valid ZIP code or city found in the query."

# # Initialize ChatOpenAI
# llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# # Create a prompt template
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an AI assistant for MSI, a company specializing in surfaces. Your role is to assist users with queries about dealers, products, and basic information. Remember to maintain context from previous messages in the conversation."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}")
# ])

# @app.route('/chat', methods=['POST', 'GET'])
# def chat_endpoint():
#     if request.method == 'POST':
#         data = request.get_json()
#         user_question = data.get('question')
#         session_id = data.get('session_id', 'default')
#     elif request.method == 'GET':
#         user_question = request.args.get('question')
#         session_id = request.args.get('session_id', 'default')
#     else:
#         return jsonify({"error": "Invalid request method"}), 405

#     if not user_question:
#         return jsonify({"error": "Question is missing"}), 400

#     # Load the conversation history
#     chat_history = load_memory(session_id)

#     # Convert the loaded history to Message objects
#     messages = [
#         SystemMessage(content="You are an AI assistant for MSI, a company specializing in surfaces. Your role is to assist users with queries about dealers, products, and basic information. Remember to maintain context from previous messages in the conversation.")
#     ]
#     for msg in chat_history:
#         if msg['role'] == 'human':
#             messages.append(HumanMessage(content=msg['content']))
#         elif msg['role'] == 'ai':
#             messages.append(AIMessage(content=msg['content']))

#     # Add the new user message
#     messages.append(HumanMessage(content=user_question))

#     # Get AI's response
#     response = llm(messages)

#     # Check if dealer information is requested
#     if "dealer" in user_question.lower() or "location" in user_question.lower():
#         dealer_info = ExtractDealerInfo(user_question)
#         response.content += f"\n\nHere's some additional dealer information: {dealer_info}"

#     # Add AI's response to the chat history
#     chat_history.append({"role": "human", "content": user_question})
#     chat_history.append({"role": "ai", "content": response.content})

#     # Save the updated chat history
#     save_memory(session_id, chat_history)

#     return jsonify({"response": response.content, "session_id": session_id})

# if __name__ == "__main__":
#     app.run(debug=False, port=5000, host='0.0.0.0')
