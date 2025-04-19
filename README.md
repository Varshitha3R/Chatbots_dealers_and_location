
# 📍 ZipCode Location Q&A Assistant with Chatbot Memory

An intelligent Flask-based web application that helps users retrieve relevant location-based information using semantic search powered by **OpenAI embeddings**, **LangChain**, and **LangGraph**. This application maintains session memory, allowing it to remember past interactions across different queries, providing a more personalized and context-aware experience.

---

## 🚀 Features

- 🔍 **Semantic Search** using cosine similarity
- 🤖 **GPT-4 Powered Answers** using OpenAI API
- 📁 Uses CSV file (`loc_emb.csv`) for location content + vector embeddings
- 🧠 Integrates **LangChain's OpenAIEmbeddings**
- 🌐 CORS enabled — ready for frontend integration
- 💬 Interactive POST endpoint for query-response
- 🧑‍💻 Simple HTML front end (`location.html`)
- 🧠 **Session Memory** with LangGraph to remember past user interactions
- 🔑 **Session Management** with `session_id` for personalized interaction

---

## 📂 Project Structure

```
zipcode/
│
├── app.py               # Flask backend
├── location.html        # Frontend (can be extended)
├── loc_emb.csv          # Location content + precomputed embeddings
├── langraph.py          # Chatbot memory and session management (LangGraph)
├── .env                 # Environment variables (e.g., OpenAI key)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/zipcode-assistant.git
cd zipcode-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your API Key

Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the App

```bash
python app.py
```

App will be live at: [http://localhost:5000](http://localhost:5000)

---

## 🔌 API Endpoints

### `POST /chat`

Get a smart location-based answer while maintaining session memory.

**Request:**

```json
{
  "question": "Tell me about the Atlanta distribution center.",
  "session_id": "optional-session-id"
}
```

**Response:**

```json
{
  "Answer": "<b>Location:</b> Atlanta, Georgia <ul><li>Details...</li></ul>",
  "Similarity": 0.89
}
```

The `session_id` is used to maintain conversation history and context. If provided, the chatbot will reference previous interactions for a more personalized response.

---

### `GET /chat`

Currently not implemented — placeholder for session handling and retrieving past interactions.

---

## 💬 Chatbot Memory with LangGraph

The chatbot leverages **LangGraph** for **session memory**. Each user query is associated with a `session_id`, which allows the bot to:

- Remember previous queries
- Provide context-aware responses across different sessions
- Personalize interactions by referencing past user behavior

`langraph.py` is responsible for managing the memory and ensuring context is maintained.

---

## 📋 Notes

- `loc_emb.csv` must contain two fields: `Content` and `embedding`
- Embeddings must be stored in stringified list format
- OpenAI GPT-4 and embedding API usage require valid API key
- Responses are formatted using basic HTML (`<b>`, `<ul>`, `<li>`, `<i>`)
- **Session Memory** relies on LangGraph, and it uses the `session_id` to manage conversation history.

---

---

## 🙌 Contact

**Author:** Varshitha R
📧 varshitharavi315@gmail.com
🔗 [[LinkedIn](https://www.linkedin.com/in/your-profile)  ](https://www.linkedin.com/in/varshithar31/)


---
```

