# Research Intelligence Assistant 🔬🤖

A Retrieval-Augmented Generation (RAG) application that allows users to ask natural-language questions about research papers and receive context-based answers with source citations.

The project combines semantic search, vector storage, and a locally running Large Language Model (LLM) to retrieve relevant research passages and generate answers grounded in the available documents.

## ✨ Features

- **PDF ingestion:** Extract and process research-paper content.
- **Text chunking:** Split extracted documents into smaller passages for retrieval.
- **Semantic search:** Generate text embeddings using Sentence Transformers.
- **Vector database:** Store and retrieve document embeddings using ChromaDB.
- **Local LLM inference:** Generate answers using Ollama and Llama 3.2.
- **Source citations:** Associate answers with retrieved sources and passages.
- **Document-specific queries:** Ask questions about a particular research paper.
- **Evaluation framework:** Measure retrieval performance, answer relevance, grounding heuristics, citations, and latency.
- **Unanswerable-question handling:** Test whether the system abstains when the available passages do not provide sufficient evidence.

## 🛠️ Tech Stack

| Component            | Technology                               |
| -------------------- | ---------------------------------------- |
| Programming language | Python                                   |
| Embedding model      | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database      | ChromaDB                                 |
| Local LLM runtime    | Ollama                                   |
| Language model       | `llama3.2:3b`                            |
| PDF processing       | PyMuPDF                                  |
| Evaluation           | Custom Python evaluation framework       |

## 🏗️ Architecture

```text
Research Paper PDFs
        |
        v
PDF Text Extraction
        |
        v
Text Cleaning and Chunking
        |
        v
Embedding Generation
        |
        v
ChromaDB Vector Database
        |
        v
User Question
        |
        v
Semantic Retrieval
        |
        v
Relevant Context + Source Metadata
        |
        v
Ollama (Llama 3.2)
        |
        v
Context-Based Answer with Citations
```

## 📂 Project Structure

```text
Research Intelligence Assistant/
│
├── ingestion.py
├── built_index.py
├── config.py
├── evaluation.py
├── eval_set_example.json
├── rag_pipeline.py
├── test_rag.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── data/                 # Local research-paper PDFs
├── output/               # Generated chunks
│   └── chunks.json
├── chromaDB/             # Persistent vector database
├── eval_results/         # Evaluation reports
├── .venv/                # Python virtual environment
└── __pycache__/          # Python cache
```

*Note: Local PDFs, the virtual environment, generated indexes, and cache files are not intended to be committed to GitHub. The exact files included depend on your `.gitignore`.*

## ⚙️ Prerequisites

Before running the project, install:

- Python 3.10 or a compatible version for the project's dependencies
- Git
- Ollama
- The `llama3.2:3b` model

The embedding model is downloaded through Sentence Transformers when it is not already available locally.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd "Research Intelligence Assistant"
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL.

### 2. Create a virtual environment

**Windows:**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from [ollama.com](https://ollama.com/).

Download the model:

```cmd
ollama pull llama3.2:3b
```

Verify that the model is available:

```cmd
ollama list
```

Make sure the Ollama service is running before starting the application.

### 5. Configure environment variables

Create a local `.env` file based on `.env.example` if your application loads configuration from that file.

Example:

```dotenv
LLM_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
RAG_TOP_K=4
RAG_MAX_DISTANCE=0.55
```

Do not commit `.env` or any API keys, passwords, or other secrets.

### 6. Add research papers

Place the research-paper PDFs in the local `data/` directory.

Only use documents you have permission to process and redistribute. The PDFs are not included in this repository by default.

### 7. Build the index

Run the ingestion and indexing scripts according to their implementation:

```cmd
python ingestion.py
python built_index.py
```

These commands are examples based on the current filenames. Confirm the scripts' responsibilities and expected execution order before running them. If the repository already includes an index-building workflow, follow that workflow rather than creating duplicate indexes.
## 🤖 Run the RAG Pipeline

The RAG pipeline retrieves relevant passages from the ChromaDB vector database and uses the local Llama 3.2 model through Ollama to generate an answer.

### Run the pipeline

Activate your virtual environment and execute:

```cmd
python rag_pipeline.py
```

If your pipeline uses a different filename or has a different entry point, replace the filename with the appropriate script.

### Example questions

Once the pipeline starts, try questions such as:

- What problems does RAG solve for LLMs?
- Explain Naive RAG versus Advanced RAG.
- What does Research_paper_1.pdf say about hallucinations?
- What are the limitations of RAG?

### How it works

1. Accepts a natural-language question.
2. Converts the question into an embedding.
3. Retrieves relevant passages from ChromaDB.
4. Passes the retrieved context to the local LLM.
5. Generates a context-based answer with source citations.
6. Abstains when the available evidence is insufficient.

**Note:** The ChromaDB index must be built before running the pipeline, and Ollama must be running with the `llama3.2:3b` model available locally.


## 🧪 Evaluation

The project includes an evaluation script and a sample evaluation dataset.

Run retrieval-only evaluation:

```cmd
python evaluation.py --retrieval-only
```

Run retrieval and generation evaluation:

```cmd
python evaluation.py
```

The evaluation framework measures:

- Hit\@k
- Mean Reciprocal Rank (MRR)
- Precision\@k
- Top-ranked source accuracy
- Retrieval latency
- Answer relevance
- Expected-keyword coverage
- Faithfulness heuristic
- Citation coverage and precision heuristics
- Correct abstention on unanswerable questions
- Time to first token and total response latency
- Approximate generation speed

Evaluation reports are saved in the local `eval_results/` directory.

### Preliminary evaluation results

The initial test used **five questions** with `llama3.2:3b`.

| Metric                        |  Initial result |
| ----------------------------- | --------------: |
| Hit\@k                        |            100% |
| MRR                           |           1.000 |
| Precision\@k                  |           66.7% |
| Top-ranked source accuracy    |            100% |
| Answer relevance              |           0.656 |
| Expected-keyword coverage     |            100% |
| Faithfulness heuristic        |            100% |
| Citation coverage             |            100% |
| Citation precision heuristic  |            100% |
| Correct end-to-end abstention |            100% |
| Median total response time    |    44.5 seconds |
| Approximate generation speed  | 5 tokens/second |

**Important:** These are preliminary results from a small evaluation set. Keyword coverage, embedding similarity, and the current faithfulness and citation checks are heuristics; they do not establish factual correctness or guarantee that every claim is supported by its citation. Some answers remain incomplete, and retrieval thresholds and latency measurements require further investigation.

## ⚠️ Current Limitations

- Local LLM inference can be slow on CPU-based hardware.
- Retrieval quality depends on document extraction, chunking, embeddings, and similarity thresholds.
- Retrieved passages may be relevant to the topic without containing enough evidence to answer the question.
- Generated answers may be incomplete even when retrieval metrics look strong.
- Citation and faithfulness scores require more rigorous validation.
- The evaluation dataset needs to be expanded and manually reviewed.
- Performance may vary with the research papers, hardware, and model configuration.

## 🔮 Future Improvements

- Expand the evaluation dataset to 30–50 or more manually reviewed questions.
- Improve retrieval relevance filtering and document-specific retrieval.
- Evaluate alternative chunk sizes and retrieval strategies.
- Improve answer completeness and evidence-based abstention.
- Introduce more rigorous claim-level citation verification.
- Investigate reranking and hybrid retrieval.
- Optimize local inference latency.
- Add structured logging and failure-case analysis.
- Develop a more polished user interface and deployment workflow.

## 🎯 Project Objective

The objective is to build a research assistant that helps users explore multiple research papers through natural-language questions, retrieve relevant evidence, and generate transparent, source-supported responses.

## 👩‍💻 Technologies and Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Natural Language Processing
- Text embeddings and semantic similarity
- Vector databases
- Local Large Language Models
- PDF ingestion and document chunking
- Evaluation and error analysis
- Python application development


**Project status:** Working prototype with preliminary evaluation. Improvements to answer completeness, retrieval quality, and evaluation reliability are ongoing.
