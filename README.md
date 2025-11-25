# The NVidiaRAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The NVidiaRAG is a sophisticated Retrieval-Augmented Generation (RAG) system designed to answer questions based on a private knowledge base of documents. It leverages state-of-the-art NVIDIA language models for generating embeddings and reranking search results, ensuring high accuracy and relevance. The entire system is wrapped in a user-friendly web interface built with Streamlit, and it uses an Oracle Autonomous Database (23ai) for efficient vector storage and retrieval.

![image](https://github.com/mamorett/the-NVidiaRAG/blob/main/infograph.png)

This project provides an end-to-end solution for building a powerful question-answering system that can be used for a variety of applications, such as customer support, internal knowledge management, and research.

## Features

- **High-Quality Embeddings:** Utilizes `nvidia/llama-3.2-nv-embedqa-1b-v2` for generating dense vector embeddings of documents and queries.
- **Advanced Reranking:** Employs `nvidia/llama-3.2-nv-rerankqa-1b-v2` to rerank retrieved documents, significantly improving the relevance of the context provided to the language model.
- **Vector Storage:** Integrates with Oracle Autonomous Database (23ai) for scalable and efficient vector storage and similarity search.
- **Web Interface:** A user-friendly Streamlit application for uploading documents, managing the knowledge base, and asking questions.
- **Document Management:** Easily upload PDF documents, which are automatically processed, chunked, and stored in the vector database.
- **Detailed Process Insights:** The Streamlit interface provides a detailed view of the RAG process, including the retrieved documents, their similarity scores, and the reranked results.
- **Optimized for Performance:** The application is optimized for performance, with features like batch processing and GPU memory management.
- **Observability:** Integrated with Langfuse for detailed tracing and monitoring of the RAG pipeline, from embedding and reranking to the final generation step.

## How it Works

The NVidiaRAG system follows a two-stage process to answer questions:

1.  **Retrieval:**
    - When a user asks a question, the system first encodes the query into a high-dimensional vector using the NVIDIA embedding model.
    - It then performs a similarity search in the Oracle vector database to retrieve the most relevant document chunks from the knowledge base.

2.  **Reranking and Generation:**
    - The retrieved document chunks are then passed to the NVIDIA reranking model, which reorders them based on their relevance to the query.
    - The top-ranked documents are combined to form a context.
    - This context is then provided to a large language model (LLM) along with the original question, and the LLM generates a comprehensive answer based on the provided information.

This two-stage process ensures that the generated answers are not only relevant but also accurate and grounded in the information contained in the knowledge base.

## Observability with Langfuse

This project uses [Langfuse](https://langfuse.com/) for tracing and observability. Langfuse allows you to monitor the performance of the RAG pipeline, track costs, and debug issues by providing a detailed view of each step in the process.

- **Traceability:** Every query is tracked as a "trace," which includes all the steps from embedding the query to generating the final answer.
- **Spans:** Each major operation (embedding, reranking, retrieval, LLM call) is tracked as a "span" within a trace, allowing you to see the duration and metadata of each step.
- **Cost Tracking:** The Langfuse OpenAI wrapper automatically tracks token usage and costs for each LLM call.
- **Metadata:** Additional metadata, such as model names, similarity scores, and reranking results, is logged to provide deeper insights into the process.

To use Langfuse, you will need to set up the environment variables listed in the "Setup and Installation" section.

## Tech Stack

- **Backend:** Python
- **Machine Learning:**
  - `PyTorch`
  - `transformers`
  - `sentence-transformers`
- **NVIDIA Models:**
  - Embedding: `nvidia/llama-3.2-nv-embedqa-1b-v2`
  - Reranking: `nvidia/llama-3.2-nv-rerankqa-1b-v2`
- **Frontend:** `Streamlit`
- **Database:** `Oracle Autonomous Database (23ai)`
- **Document Processing:** `PyPDFLoader`, `LangChain`
- **Observability:** `Langfuse`

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/the-NVidiaRAG.git
    cd the-NVidiaRAG
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    - Create a `.env` file in the root of the project.
    - Add the following environment variables to the `.env` file:
      ```
      # Oracle Database
      ORACLE_USER="your_oracle_username"
      ORACLE_PASSWORD="your_oracle_password"
      ORACLE_DSN="your_oracle_dsn"

      # OpenAI
      OPENAI_API_KEY="your_openai_api_key"
      OPENAI_BASE_URL="your_openai_base_url" # Optional
      OPENAI_MODEL="your_openai_model" # Optional

      # Langfuse
      LANGFUSE_SECRET_KEY="your_langfuse_secret_key"
      LANGFUSE_PUBLIC_KEY="your_langfuse_public_key"
      LANGFUSE_HOST="https://cloud.langfuse.com" # Or your self-hosted instance
      ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run therag.py
    ```

2.  **Upload documents:**
    - Open the application in your web browser.
    - Use the file uploader in the sidebar to upload your PDF documents.
    - Click the "Process & Add to Database" button to start the ingestion process.

3.  **Ask a question:**
    - Once the documents have been processed, you can ask questions in the main text area.
    - Click the "Search & Answer" button to get an answer.
    - The application will display the generated answer, as well as detailed information about the retrieval and reranking process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Mattia Moretti