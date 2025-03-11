from flask import Flask, request, jsonify
import re
import requests
from bs4 import BeautifulSoup
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from flask_cors import CORS
import uuid

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize OpenAI client using your API key
llm = ChatOpenAI(openai_api_key="", model="gpt-4")

# Store session memories in a dictionary
session_memory_store = {}

# Function to create session memory
def create_session_memory():
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    session_memory_store[session_id] = memory
    return session_id, memory

# Function to process the main user query
def process_main_query(user_query, session_id, memory):
    # Handle basic queries like "Hi"
    if user_query.lower() == "hi":
        return {"response": f"Hello! How can I help you today? You can ask about dealers.",
            }

    # Try extracting location (ZIP or city) from the query
    extracted_location = extract_location_from_query(user_query)

    # If no location is found, check if user asked for a specific dealer by name
    if not extracted_location:
        dealer_info = scrape_dealer_information_by_name(user_query)
        if dealer_info:
            # Now check if the query is asking for a specific detail like "services", "location", etc.
            if "products" in user_query.lower() or "services" in user_query.lower():
                services = dealer_info.get("Services", "No services found")
                return {"response": f"The products and services offered by {user_query} are: {services}", "session_id": session_id}
            elif "location" in user_query.lower():
                location = dealer_info.get("Location", "Location information not available.")
                return {"response": f"The location of {user_query} is: {location}", "session_id": session_id}
            elif "contact" in user_query.lower():
                contact = dealer_info.get("Contact", "Contact information not available.")
                return {"response": f"Contact information for {user_query} is: {contact}", "session_id": session_id}
            else:
                # Default case if no specific detail is requested
                return {"response": f"Dealer {user_query} information: {dealer_info}", "session_id": session_id}
        else:
            return {"response": "No valid dealer information found based on the query.", "session_id": session_id}

    # If location is found, scrape dealer info for that location
    dealers_info = scrape_dealer_information(extracted_location)
    if dealers_info:
        context = f"Dealer information for {extracted_location}: {dealers_info}"

        system_role = f"""You are an AI assistant with the ability to analyze and provide answers based on the dealer information scraped from the provided ZIP code. Answer the question clearly and organize the information in bullet points for easy readability.

        Context: {context}

        Format the response as follows:
        1. List each dealer found in the ZIP code area.
        2. For each dealer, include:
            - Dealer Name
            - Location (city, state, etc.)
            - Products/Services they offer (e.g., countertops, flooring, etc.)
            - Any additional details (e.g., contact info, operating hours, etc.)
        3. Make sure the response is easy to read and use bullet points to clearly separate each dealer's information.
        4. Scrape all the dealer locations from the Zipcode
        5. Scrape all the dealers' locations near from the Zipcode

        You are an AI assistant for Dealer location info.
        Provide information based on the given context only.
        Consider past conversation for better responses.
        Keep the responses in HTML format - use only these tags: <li>, <ol>, <p>, <b>, <i>.

        Chat History: {memory.load_memory_variables({})["history"]}

        Output Format -> Response: <response>"""

        # Add the system message to the conversation memory
        memory.chat_memory.add_message(SystemMessage(content=system_role))

        # Create the ConversationChain with session-specific memory
        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )

        # Generate response from LangChain
        response = conversation_chain.predict(input=user_query)

        # Append the conversation to session memory
        memory.chat_memory.add_message({"role": "user", "content": user_query})
        memory.chat_memory.add_message({"role": "assistant", "content": response})

        # Return the response along with session_id
        return {"response": response, "session_id": session_id}
    else:
        return {"response": "No dealer information found for the given location.", "session_id": session_id}

# Function to extract ZIP code or city from the query
def extract_location_from_query(user_query):
    zip_pattern = r"\b\d{5}\b"  # This pattern matches 5-digit ZIP codes
    city_pattern = r"\b([A-Za-z]+(?:\s[A-Za-z]+)*)\b"  # This pattern matches city names

    zip_match = re.search(zip_pattern, user_query)
    city_match = re.search(city_pattern, user_query)

    if zip_match:
        return zip_match.group()  # Return ZIP code if found
    elif city_match:
        return city_match.group()  # Return city name if found
    else:
        return None  # No ZIP code or city found

# Function to scrape dealer information by name
def scrape_dealer_information_by_name(dealer_name):
    # We are now dynamically searching for the dealer name on the website.
    url = f"https://www.msisurfaces.com/dealer-locator/countertops-flooring-hardscaping-stile/{dealer_name.replace(' ', '-').lower()}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        dealer_data = {}

        # Find all the dealer names on the page
        for h4_tag in soup.find_all('h4'):
            b_tag = h4_tag.find('b')
            if b_tag and dealer_name.lower() in b_tag.get_text(strip=True).lower():
                # Collect dealer information if it matches the name
                dealer_data['Dealer Name'] = b_tag.get_text(strip=True)

                # Check if there's a div with location and services details
                div_tag = h4_tag.find_next('div')
                if div_tag:
                    span_tags = div_tag.find_all('span')
                    if span_tags:
                        location = span_tags[0].get_text(strip=True)
                        services = [span.get_text(strip=True) for span in span_tags[1:]]
                        dealer_data['Location'] = location
                        dealer_data['Services'] = services
                    else:
                        dealer_data['Services'] = "No additional services found."
                else:
                    dealer_data['Services'] = "No additional services found."

        if dealer_data:
            return dealer_data
        else:
            return None
    else:
        return None

# Function to scrape dealer information for the given location
def scrape_dealer_information(location):
    url = f"https://www.msisurfaces.com/dealer-locator/countertops-flooring-hardscaping-stile/{location}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        dealers_info = []

        # Scrape all dealers listed in the location
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

# Define an API route to handle queries
@app.route('/ask', methods=['GET', 'POST'])
def ask():
    user_query = request.args.get("query", "")
    session_id = request.args.get("session_id", "")

    if not session_id:
        # Create new session if no session ID provided
        session_id, memory = create_session_memory()
    else:
        # Retrieve existing memory for the session
        memory = session_memory_store.get(session_id)
        if not memory:
            # If session expired or invalid, create new session
            session_id, memory = create_session_memory()

    if user_query:
        # Process the query using the session-specific memory
        response_data = process_main_query(user_query, session_id, memory)
        return jsonify(response_data)
    else:
        return jsonify({"response": "No query provided.", "session_id": session_id})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)
