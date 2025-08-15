## 📖 Intelligent Hafez Verse Search

A Streamlit-based web application enabling semantic search over the complete divan of Hafez using state-of-the-art Persian sentence embeddings and vector search with Qdrant.

---

### 🚀 Project Overview

This project provides an intelligent search interface for Hafez’s ghazals (poems). It ingests the full text of *دیوان حافظ* (divan of Hafez), splits it into couplets (بیت), computes embeddings via the `heydariAI/persian-embeddings` SentenceTransformer model, and stores the data in a Qdrant vector database for fast, approximate nearest neighbor search. Users can enter any Persian query to retrieve the most semantically relevant verses.

### ✨ Key Features

* **Semantic Search**: Finds verses by meaning, not just keyword matching.
* **Persian Language Support**: Full right-to-left (RTL) layout and Persian font (Vazirmatn).
* **Relevancy Scores**: Displays a distance score for each result.
* **Dockerized**: Easily containerized for consistent deployment.
* **Interactive UI**: Built with Streamlit for rapid development and sharing.

### 📂 Repository Structure

```bash
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Service orchestration
├── LICENSE                    # MIT License
├── README.md                  # This documentation
├── requirements.txt           # Python dependencies
├── hafez.txt                  # Raw divan text (source)
├── run_embedding.py           # Script to parse and embed verses
├── embed_query_tester.py      # (Optional) Embedding & query testing script
├── query.py                   # Programmatic search interface
├── streamlit_app.py           # Streamlit web application
└── docker-compose.yml         # Includes Qdrant service for vector storage
```

### 🛠️ Prerequisites

* Python 3.12+
* Docker & Docker Compose (for containerized setup)
* Internet access to download the `heydariAI/persian-embeddings` model

### 🔧 Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/bachelor-project.git
   cd bachelor-project
   ```

2. **Create a Python virtual environment (optional but recommended)**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Generate embeddings**:

   ```bash
   python run_embedding.py
   ```

   This script reads `hafez.txt`, parses ghazals and verses, computes embeddings, and uploads them to the Qdrant collection.

### ▶️ Running the Application Locally

```bash
streamlit run streamlit_app.py --server.port=7777 --server.address=0.0.0.0
```

Open your browser at `http://localhost:7777` to use the search interface.

### 🐳 Dockerized Deployment

1. **Build the Docker image**:

   ```bash
   docker build -t hafez-search .
   ```

2. **Run with Docker Compose**:

   ```bash
   docker-compose up --build
   ```

This binds port `7777` on the host and mounts the current directory to allow live edits.

### Use the Makefile
or instead of all these complications you could use the Makefile we have and run

```bash
make build && make up
```

### 🧩 Usage Examples

* **Basic search**: Enter keywords like `عشق` or full phrases like `آزادگی`.
* **Conceptual query**: `سفر در زمان` to find verses about journeys and time.

### 🌐 Programmatic Access

Use `query.py` to perform searches in Python:

```python
from query import search
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('heydariAI/persian-embeddings')
vec = model.encode(['عشق'])[0]
results = search(vec)
print(results)
```

### 📖 Customization

* **Model**: Swap out `heydariAI/persian-embeddings` for any compatible SentenceTransformer.
* **Data Source**: Replace `hafez.txt` with another Persian corpus.
* **UI**: Modify `streamlit_app.py` and CSS under `st.markdown` for branding.

---

*Developed for Bachelor’s Project by Alireza Khorami*

for the non technical document, take a look at this:

https://docs.google.com/document/d/1kPBD21pEvcRnMVVYnimJUtyz1Lpe7ykyGnAyOn7jg5w/edit?usp=sharing 
### Research quickstart
- Run `pip install -r requirements.txt && pip install -r requirements.research.txt`
- Add queries/judgments in `data/gold/`
- Run `make -f Makefile.research eval`
- See `RESULTS.md` and `paper/mini-study.md`
