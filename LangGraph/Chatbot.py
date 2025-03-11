from IPython.display import Image,display
from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
import requests
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode,tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import csv
import os
import uuid
import numpy as np
import openai  # Direct OpenAI API integration
from flask import Flask, request, jsonify
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from flask_cors import CORS  # Import CORS
from sklearn.metrics.pairwise import cosine_similarity
import csv
import numpy as np
from typing import List, Dict
import uuid
import re
import requests
from bs4 import BeautifulSoup
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.schema import SystemMessage
import re
import requests
from flask import Flask, request, jsonify
import json
app = Flask(__name__)


class State(TypedDict):
  messages:Annotated[list,add_messages]

# embedding_model = OpenAIEmbeddings(api_key="sk-proj-")

import csv
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai


# Function to process CSV data
def multiply(a: int, b: int) -> int:
  """Multiplies two integers and returns the result."""
  return a * b


def add(a: int, b: int) -> int:
  """Adds two integers and returns the result."""
  return a + b

# Function to fetch dealer location from a query
def FetchDealerLocation(question):
    """Please extract zipcode or address or location or city or state from the following question and if you don't find it tell i don't know""" # Added docstring here
    system_role = "Please extract zipcode or address or location or city or state from the following question and if you don't find it tell i don't know\n\n Question:" + question + "?"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "assistant", "content": system_role},
            {"role": "user", "content": "UserInput: " + question}
        ],
        temperature=0
    )

    address = response.choices[0].message.content
    print(address)
    return address

# Function to scrape dealer information from a location
def ScrapeDealerInfo(location):
    """Scrape dealer information from a location""" # Added docstring here
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

# Function to extract dealer information from a question
def ExtractDealerInfo(question):
  """Extract dealer information from a question""" # Added docstring here
  extracted_location = FetchDealerLocation(question)

  if extracted_location:
        # Scrape the dealer information for the location
        dealers_info = ScrapeDealerInfo(extracted_location)

        if dealers_info:
            context = f"Dealer information for {extracted_location}: {dealers_info}"

            system_role = f"""You are an MSI AI assistant with the ability to analyze the context and provide answers based on the dealer information scraped from the provided ZIP code.
            Answer the question clearly and organize the information in bullet points for easy readability.
            Keep the responses in HTML format - use only these tags: <li>, <ol>, <p>, <b>, <i>.
            Context: {context}"""

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "assistant", "content": system_role},
                    {"role": "user", "content": "UserInput: " + question}
                ],
                temperature=0
            )

            result = response.choices[0].message.content
            return result
        else:
            return "No dealer information found for the given location."
  else:
        return "No valid ZIP code or city found in the query."

session_memory = {}

def getSessionMemory(sessionID):
    """Retrieve or initialize session-specific memory."""
    if sessionID not in session_memory:
        session_memory[sessionID] = ConversationBufferMemory(k=10, memory_key="chat_history", return_messages=True)
    return session_memory[sessionID]


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0,api_key="", model_name="gpt-4o-mini")

llm.invoke("hello").content

tools =[ ScrapeDealerInfo,ExtractDealerInfo,multiply, add]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state:State):
  return {"messages":[llm_with_tools.invoke(state["messages"])]}

from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

graph_builder = StateGraph(State)

#define nodes
graph_builder.add_node("assistant",chatbot)
graph_builder.add_node("tools",ToolNode(tools))

#define edges
graph_builder.add_edge(START,"assistant")
graph_builder.add_conditional_edges("assistant", tools_condition)
graph_builder.add_edge("tools","assistant")
react_graph = graph_builder.compile()

react_graph = graph_builder.compile(checkpointer=memory)
# display(Image(react_graph.get_graph().draw_mermaid_png()))

response =react_graph.invoke({"messages":[HumanMessage(content="what is the weather in bengaluru . Multiply it by 2 and add 5")]})
print(response["messages"])

for m in response["messages"]:
    m.pretty_print()

@app.route('/chat', methods=['POST', 'GET'])
def chat_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        user_question = data.get('question')
        session_id = data.get('session_id', 'default')
        if not user_question:
          return jsonify({"error": "Question is missing"}), 400

        memory = getSessionMemory(session_id)

        response = react_graph.invoke(
            {"messages": [HumanMessage(content=user_question)]},
           config={"configurable": {"thread_id": session_id}}
        )
        return jsonify({"response": response["messages"][-1].content})

    elif request.method == 'GET':
        user_question = request.args.get('question')
        session_id = request.args.get('session_id', 'default')
        if not user_question:
           return jsonify({"error": "Question is missing"}), 400

        memory = getSessionMemory(session_id)

        response = react_graph.invoke(
            {"messages": [HumanMessage(content=user_question)]},
            config={"configurable": {"thread_id": session_id}}
        )
        return jsonify({"response": response["messages"][-1].content})



