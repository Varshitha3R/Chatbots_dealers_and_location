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

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and OpenAI API key from .env
app = Flask(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY")  # Get the API key from the .env file

if not openai_api_key:
    raise ValueError("API Key is missing. Please add it to the .env file.")

openai.api_key = openai_api_key  # Set the OpenAI API key

# Initialize OpenAI embeddings model
embedding_model = OpenAIEmbeddings(api_key=openai_api_key)

# Enable CORS for all routes
CORS(app)

# Function to process CSV data and return locations and embeddings
def process_csv_data(csv_file="loc_emb.csv"):
    """Process the CSV file and return a list of dictionaries containing 'Content' and 'embedding'."""
    locations = []
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert the embedding string to a numpy array
            embedding = np.array([float(x) for x in row['embedding'].strip('[]').split(',')])
            locations.append({
                'Content': row['Content'],
                'embedding': embedding.tolist()  # Convert to list to make it JSON serializable
            })
    return locations

# Function to get the embedding of a query
def get_query_embedding(query):
    """Generate embedding for the user query."""
    return embedding_model.embed_query(query)

# Function to match location with query embeddings using cosine similarity
def match_location_with_embeddings(query_embedding, csv_file="loc_emb.csv"):
    """Match the query embedding with embeddings from the CSV using cosine similarity."""
    locations = process_csv_data(csv_file)
    similarities = []

    for loc in locations:
        # Compute cosine similarity between query embedding and each embedding in the CSV
        similarity = cosine_similarity([query_embedding], [loc['embedding']])[0][0]
        similarities.append((loc, similarity))

    # Sort the results based on similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the most similar location (or top n)
    return similarities[0] if similarities else None

# Function to generate focused answer using OpenAI's GPT-4 API
def generate_focused_answer(question, matched_content):
    """Generate a focused answer using OpenAI GPT-4 model."""
    try:
        # Construct the system message with the matched content
        system_role = f"""
        You are an intelligent assistant powered by GPT-4. Your role is to help users find location-related information based on their questions.
        You have access to a set of content related to different locations, stored as embeddings in a CSV file.

        Instructions:
        - Understand the user’s question and identify the key concepts or topics.
        - Match the user’s query with the most relevant content from the CSV data.
        - Once you find the matching content, extract relevant information and provide an answer based on that content.
        - Structure your responses clearly and provide concise, informative answers.
        - Structure your responses clearly, in bullet points using <ul> and <li> tags for lists.
        - Use <b> tags for bold text where appropriate (for headings or important sections).
        - Use <i> tags for italicizing relevant content if needed.
        - Avoid providing any information that is not present in the provided content or CSV.
        - Ensure that the response is clear, informative, and answers the user’s question as precisely as possible.
        - If there is no matching location in the data, inform the user politely and suggest rephrasing their query.
        - If user asks about U.S. Sales and Distribution Centers , try to give these country with the response (Here are the cities in a line:

        Atlanta, Georgia; Austin, Texas; Baltimore, Maryland; Boston, Massachusetts; Charlotte, North Carolina; Chicago, Illinois; Cincinnati, Ohio; Cleveland, Ohio; Columbus, Ohio; Dallas, Texas; Deerfield Beach, Florida; Denver, Colorado; Detroit, Michigan; Dulles, Virginia; Edison, New Jersey; Grand Rapids, Michigan - Coming Soon; Houston, Texas; Indianapolis, Indiana; Jacksonville, Florida - Coming Soon; Kansas City, Kansas; Las Vegas, Nevada; Long Island, New York; Los Angeles, California; Milwaukee, Wisconsin; Minneapolis, Minnesota; Nashville, Tennessee; New Haven, Connecticut; Oklahoma City, Oklahoma; Omaha, Nebraska; Orlando, Florida; Philadelphia, Pennsylvania; Phoenix, Arizona; Pittsburgh, Pennsylvania; Portland, Oregon; Raleigh, North Carolina; Richmond, Virginia; Rochester, New York; Sacramento, California; Salt Lake City, Utah; San Antonio, Texas; San Diego, California; San Francisco Bay Area, California; Savannah, Georgia; Seattle, Washington; St. Louis, Missouri; Tampa Bay, Florida; Virginia Beach, Virginia.)

        Context: {matched_content}  # Insert the relevant content here
        """

        # Use OpenAI's Chat API (GPT-4)
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Specify the GPT-4 model
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": f"Given the following content: '{matched_content}', answer this question: '{question}'"}
            ],
            temperature=0  # Set temperature to 0 for deterministic answers
        )

        # Return the model's response
        return response['choices'][0]['message']['content']

    except Exception as e:
        return f"Error occurred: {str(e)}"


# Function to integrate the entire process: find the location and return related data from the CSV
def get_location_info_from_question(question, session_id, csv_file="loc_emb.csv"):
    # Handle basic greetings first
    greetings = ["hi", "hello", "hey", "greetings", "howdy", "good morning", "good evening", "good afternoon"]
    if any(greeting in question.lower() for greeting in greetings):
        return {"Answer": "Hello! How can I assist you today?"}

    # Get the embedding for the user query
    query_embedding = get_query_embedding(question)

    # Match the query embedding to the CSV embeddings
    matched_location = match_location_with_embeddings(query_embedding, csv_file)

    if matched_location:
        # Extract the content from the matched location
        matched_content = matched_location[0]['Content']

        # Generate a focused answer based on the query and the matched content
        focused_answer = generate_focused_answer(question, matched_content)

        return {"Answer": focused_answer, "Similarity": matched_location[1]}  # Return only the focused answer
    else:
        return {"Answer": "No matching location found based on the provided query."}

# Flask route to handle user query (POST method)
@app.route('/chat', methods=['POST'])
def location_info():
    try:
        # Get the JSON data sent from the frontend
        data = request.get_json()  # This will parse the JSON body
        if data is None:
            return jsonify({"error": "Invalid JSON format."}), 400
        question = data.get('question')
        session_id = data.get('session_id', str(uuid.uuid4()))  # Generate random session ID if not provided

        if question:
            response = get_location_info_from_question(question, session_id)
            return jsonify(response)
        else:
            return jsonify({"error": "No question provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask route to retrieve conversation information (GET method)
@app.route('/chat', methods=['GET'])
def get_conversation_info():
    session_id = request.args.get('session_id')  # Retrieve session_id from URL parameters

    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400

    # Here you'd need to retrieve session memory if needed (this example doesn't include memory handling)
    return jsonify({"error": "No conversation handling set up for this route"}), 404

if __name__ == "__main__":
    app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)
