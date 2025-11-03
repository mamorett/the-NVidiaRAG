import os
import streamlit as st
from dotenv import load_dotenv
import oracledb
import numpy as np
import torch
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
import json
import array
import gc

# Import your RAG class (assuming it's in a file called rag_class.py)
from nvidiaRAGclass import RAGEmbeddingReranker
# Load environment variables
load_dotenv()

# Initialize RAG model
@st.cache_resource
def load_rag_model():
    """Load and cache the RAG embedding and reranking model."""
    return RAGEmbeddingReranker()

rag_model = load_rag_model()

def get_optimal_batch_size(num_chunks, available_memory_gb=10):
    """Calculate optimal batch size based on available GPU memory.
    
    Args:
        num_chunks: Total number of chunks to process
        available_memory_gb: Available GPU memory in GB (default 10 for safety)
    
    Returns:
        int: Optimal batch size
    """
    # Estimate: each chunk embedding takes ~50MB with NV-Embed-v2
    # With 10GB available, we can safely process 200 chunks at once
    # But let's be conservative and use 128 as max batch size
    
    estimated_batch_size = int((available_memory_gb * 1024) / 50)  # Convert GB to MB
    max_batch_size = min(128, estimated_batch_size)
    
    # Don't go below 32 unless we have very few chunks
    min_batch_size = 32
    
    if num_chunks < min_batch_size:
        return num_chunks
    
    return max(min_batch_size, min(max_batch_size, num_chunks))

def get_oracle_connection():
    """Create and return Oracle Autonomous Database connection.
    
    Returns:
        oracledb.Connection: Active database connection
    """
    try:
        user = os.getenv("ORACLE_USER")
        password = os.getenv("ORACLE_PASSWORD")
        dsn = os.getenv("ORACLE_DSN")
        
        if not all([user, password, dsn]):
            st.error("Missing Oracle connection parameters in .env file")
            return None
        
        connection = oracledb.connect(
            user=user,
            password=password,
            dsn=dsn
        )
        return connection
        
    except Exception as e:
        st.error(f"Failed to connect to Oracle: {e}")
        return None

def initialize_vector_table():
    """Initialize Oracle 23AI vector table for storing document embeddings."""
    conn = get_oracle_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT COUNT(*) FROM user_tables WHERE table_name = 'IT_DOCUMENTS'
        """)
        table_exists = cursor.fetchone()[0] > 0
        
        if not table_exists:
            # Create table with vector column for embeddings (2048 dimensions)
            create_table_sql = """
            CREATE TABLE it_documents (
                id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                content CLOB,
                metadata CLOB,
                embedding VECTOR(2048, FLOAT32),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            conn.commit()
            st.info("✓ Created IT_DOCUMENTS table")
        
        # Check if vector index exists
        cursor.execute("""
            SELECT COUNT(*) FROM user_indexes WHERE index_name = 'IT_DOCS_VEC_IDX'
        """)
        index_exists = cursor.fetchone()[0] > 0
        
        if not index_exists:
            try:
                create_index_sql = """
                CREATE VECTOR INDEX it_docs_vec_idx 
                ON it_documents(embedding)
                ORGANIZATION NEIGHBOR PARTITIONS
                DISTANCE COSINE
                WITH TARGET ACCURACY 95
                """
                cursor.execute(create_index_sql)
                conn.commit()
                st.info("✓ Created vector index")
            except Exception as e:
                st.warning(f"Could not create advanced vector index: {e}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize vector table: {e}")
        if conn:
            try:
                conn.close()
            except:
                pass
        return False

def add_documents_to_oracle_batch(chunks_batch, embeddings_batch):
    """Add a batch of document chunks to Oracle.
    
    Args:
        chunks_batch: List of chunks to add
        embeddings_batch: Corresponding embeddings (numpy array)
    
    Returns:
        bool: Success status
    """
    conn = get_oracle_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        insert_sql = """
        INSERT INTO it_documents (content, metadata, embedding)
        VALUES (:content, :metadata, :embedding)
        """
        
        # Prepare batch data
        batch_data = []
        for i, chunk in enumerate(chunks_batch):
            # Convert embedding to array format for Oracle
            embedding_array = array.array('f', embeddings_batch[i].astype(np.float32))
            
            # Convert metadata dict to JSON string
            metadata_str = json.dumps(chunk.metadata, default=str)
            
            batch_data.append({
                'content': chunk.page_content,
                'metadata': metadata_str,
                'embedding': embedding_array
            })
        
        # Execute batch insert
        cursor.executemany(insert_sql, batch_data)
        conn.commit()
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Failed to add documents batch to Oracle: {e}")
        if conn:
            try:
                conn.rollback()
                conn.close()
            except:
                pass
        return False

def retrieve_similar_documents(query, top_k=10):
    """Retrieve documents similar to the query using vector similarity search.
    
    Args:
        query (str): User query
        top_k (int): Number of documents to retrieve
    
    Returns:
        list: List of tuples (content, metadata, similarity_score)
    """
    conn = get_oracle_connection()
    if not conn:
        return []
    
    try:
        # Generate query embedding
        with torch.cuda.amp.autocast():  # Use automatic mixed precision
            query_embedding = rag_model.encode_queries([query])
        
        query_embedding_np = query_embedding.cpu().numpy()[0].astype(np.float32)
        query_array = array.array('f', query_embedding_np)
        
        # Clear references
        del query_embedding, query_embedding_np
        
        cursor = conn.cursor()
        
        # Oracle 23AI vector similarity search
        search_sql = """
        SELECT content, metadata, 
               VECTOR_DISTANCE(embedding, :query_vec, COSINE) as distance
        FROM it_documents
        ORDER BY distance
        FETCH FIRST :top_k ROWS ONLY
        """
        
        cursor.execute(search_sql, {'query_vec': query_array, 'top_k': top_k})
        
        results = []
        for row in cursor:
            content, metadata_str, distance = row
            
            # Handle CLOB content
            if hasattr(content, 'read'):
                content = content.read()
            
            # Convert distance to similarity
            similarity = 1.0 - float(distance)
            
            # Parse metadata JSON
            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
            except:
                metadata = {}
            
            results.append((content, metadata, similarity))
        
        cursor.close()
        conn.close()
        return results
        
    except Exception as e:
        st.error(f"Failed to retrieve documents: {e}")
        if conn:
            conn.close()
        return []

def get_document_count():
    """Get the total number of documents in the database.
    
    Returns:
        int: Number of documents, or -1 if error
    """
    conn = get_oracle_connection()
    if not conn:
        return -1
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM it_documents")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        if conn:
            conn.close()
        return -1

def display_answer_with_thinking(response_text):
    """Parse and display the response with styled thinking section.
    
    Args:
        response_text (str): The raw response from the LLM
    """
    # Check if there's a thinking section
    if "</think>" in response_text:
        # Split at the closing tag
        parts = response_text.split("</think>", 1)
        thinking_part = parts[0]
        answer_part = parts[1].strip() if len(parts) > 1 else ""
        
        # Remove opening <think> tag if present
        if "<think>" in thinking_part:
            thinking_part = thinking_part.split("<think>", 1)[1]
        
        # Display thinking
        with st.container():
            st.info("💭 **Thinking Process**")
            st.write(thinking_part.strip(), unsafe_allow_html=False)
        
        st.divider()
        
        # Display the actual answer
        if answer_part:
            st.success("✅ **Answer**")
            st.markdown(answer_part)
    else:
        # No thinking section, just display the answer
        st.success("✅ **Answer**")
        st.markdown(response_text)


def get_all_documents_metadata():
    """Retrieve metadata and preview of all documents in the database.
    
    Returns:
        list: List of tuples (id, content_preview, metadata, created_at)
    """
    conn = get_oracle_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        
        # Query to get document info
        query_sql = """
        SELECT id, content, metadata, created_at
        FROM it_documents
        ORDER BY created_at DESC
        """
        
        cursor.execute(query_sql)
        
        results = []
        for row in cursor:
            doc_id, content, metadata_str, created_at = row
            
            # Handle CLOB content
            if hasattr(content, 'read'):
                content = content.read()
            
            # Get preview (first 150 chars)
            preview = content[:150] + "..." if len(content) > 150 else content
            
            # Parse metadata JSON
            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
            except:
                metadata = {}
            
            results.append((doc_id, preview, metadata, created_at))
        
        cursor.close()
        conn.close()
        return results
        
    except Exception as e:
        st.error(f"Failed to retrieve documents: {e}")
        if conn:
            conn.close()
        return []


def get_unique_source_files():
    """Get a list of unique source files from document metadata.
    
    Returns:
        list: List of tuples (source_file, chunk_count, first_added)
    """
    conn = get_oracle_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        
        # Use array fetch for better performance
        cursor.arraysize = 1000
        
        # Query to get only metadata and timestamp
        query_sql = """
        SELECT DBMS_LOB.SUBSTR(metadata, 4000, 1) as metadata_substr, 
               created_at
        FROM it_documents
        ORDER BY created_at DESC
        """
        
        cursor.execute(query_sql)
        
        # Parse and aggregate by source file in Python
        sources = {}
        row_count = 0
        
        while True:
            rows = cursor.fetchmany(1000)  # Fetch in batches
            if not rows:
                break
            
            for metadata_substr, created_at in rows:
                row_count += 1
                
                try:
                    # Parse metadata (only first 4000 chars which should contain source)
                    metadata = json.loads(metadata_substr) if metadata_substr else {}
                    source = metadata.get('source', 'Unknown')
                    
                    # Extract just the filename
                    if source != 'Unknown':
                        source = os.path.basename(source)
                    
                    if source in sources:
                        sources[source]['count'] += 1
                        # Keep earliest date
                        if created_at < sources[source]['date']:
                            sources[source]['date'] = created_at
                    else:
                        sources[source] = {
                            'count': 1,
                            'date': created_at
                        }
                except Exception as e:
                    # Skip malformed metadata
                    continue
        
        # Convert to list of tuples
        results = [(source, info['count'], info['date']) 
                   for source, info in sources.items()]
        results.sort(key=lambda x: x[2], reverse=True)  # Sort by date (newest first)
        
        cursor.close()
        conn.close()
        return results
        
    except Exception as e:
        st.error(f"Failed to retrieve source files: {e}")
        import traceback
        st.error(traceback.format_exc())
        if conn:
            try:
                conn.close()
            except:
                pass
        return []


def delete_documents_by_source(source_file):
    """Delete all chunks from a specific source file.
    
    Args:
        source_file (str): Name of the source file to delete
    
    Returns:
        int: Number of documents deleted, or -1 on error
    """
    conn = get_oracle_connection()
    if not conn:
        return -1
    
    try:
        cursor = conn.cursor()
        
        # First, get all document IDs that match the source file
        select_sql = """
        SELECT id, metadata
        FROM it_documents
        """
        
        cursor.execute(select_sql)
        
        ids_to_delete = []
        for row in cursor:
            doc_id, metadata_str = row
            
            # Handle CLOB metadata
            if hasattr(metadata_str, 'read'):
                metadata_str = metadata_str.read()
            
            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
                source = metadata.get('source', '')
                
                # Extract just the filename
                if source:
                    source = os.path.basename(source)
                
                # Check if this matches the file we want to delete
                if source == source_file:
                    ids_to_delete.append(doc_id)
            except:
                continue
        
        # Delete all matching documents
        if ids_to_delete:
            delete_sql = """
            DELETE FROM it_documents
            WHERE id = :doc_id
            """
            
            for doc_id in ids_to_delete:
                cursor.execute(delete_sql, {'doc_id': doc_id})
            
            conn.commit()
            deleted_count = len(ids_to_delete)
        else:
            deleted_count = 0
        
        cursor.close()
        conn.close()
        
        return deleted_count
        
    except Exception as e:
        st.error(f"Failed to delete documents: {e}")
        if conn:
            try:
                conn.rollback()
                conn.close()
            except:
                pass
        return -1


def add_to_db(uploaded_file):  # Single file, not list
    """Process and add an uploaded PDF file to the Oracle vector database with optimized batching.
    
    Args:
        uploaded_file: Single uploaded file object to be processed
    
    Returns:
        bool: Success status
    """
    temp_file_path = None
    
    try:
        # Save the uploaded file to a temporary path
        temp_file_path = os.path.join("/tmp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        # Load the file using PyPDFLoader
        st.info(f"📖 Loading {uploaded_file.name}...")
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        if not data:
            st.warning(f"No content extracted from {uploaded_file.name}")
            return False

        # Store metadata and content
        doc_metadata = [data[i].metadata for i in range(len(data))]
        doc_content = [data[i].page_content for i in range(len(data))]

        # Split documents into smaller chunks
        st.info(f"✂️ Splitting into chunks...")
        st_text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            chunk_size=256,  # You had 100, I recommend 256
            chunk_overlap=50
        )
        st_chunks = st_text_splitter.create_documents(doc_content, doc_metadata)

        if not st_chunks:
            st.warning(f"No chunks created from {uploaded_file.name}")
            return False

        # Calculate optimal batch size based on available GPU memory
        batch_size = get_optimal_batch_size(len(st_chunks), available_memory_gb=10)
        
        st.info(f"🚀 Processing {len(st_chunks)} chunks with optimized batch size: {batch_size}")
        
        # Process chunks in optimized batches
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_batches = (len(st_chunks) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(st_chunks), batch_size):
            batch_end = min(batch_idx + batch_size, len(st_chunks))
            batch_chunks = st_chunks[batch_idx:batch_end]
            
            current_batch = (batch_idx // batch_size) + 1
            status_text.text(f"⚡ Processing batch {current_batch}/{total_batches} ({len(batch_chunks)} chunks)...")
            
            # Prepare batch texts
            chunk_texts = [chunk.page_content for chunk in batch_chunks]
            
            # Generate embeddings with mixed precision for better memory efficiency
            with torch.cuda.amp.autocast():  # Automatic Mixed Precision
                embeddings = rag_model.encode_documents(chunk_texts)
            
            # Move to CPU and convert to numpy immediately
            embeddings_np = embeddings.cpu().numpy()
            
            # Delete GPU tensor immediately
            del embeddings
            
            # Add batch to Oracle database (this happens on CPU, GPU is free)
            success = add_documents_to_oracle_batch(batch_chunks, embeddings_np)
            
            # Clean up
            del embeddings_np, chunk_texts
            gc.collect()
            
            if not success:
                st.error(f"✗ Failed to store batch {current_batch}")
                progress_bar.empty()
                status_text.empty()
                return False
            
            # Update progress
            progress = batch_end / len(st_chunks)
            progress_bar.progress(progress)
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✓ Successfully processed {uploaded_file.name} ({len(st_chunks)} chunks)")
        return True
            
    except torch.cuda.OutOfMemoryError as e:
        st.error(f"GPU Out of Memory error for {uploaded_file.name}")
        st.error(str(e))
        st.info("Try closing other GPU applications or reduce the document size.")
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return False
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False
    finally:
        # Remove the temporary file after processing
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # Cleanup
        gc.collect()


def run_rag_chain_with_details(query):
    """Process a query using RAG and return detailed information.
    
    Args:
        query (str): The user's question
    
    Returns:
        tuple: (answer, retrieval_info, rerank_info)
    """
    # Step 1: Retrieve documents
    retrieved_docs = retrieve_similar_documents(query, top_k=10)
    
    if not retrieved_docs:
        return "I couldn't find any relevant information in the database. Please add some IT documents first.", [], []
    
    # Extract content for reranking
    doc_contents = [doc[0] for doc in retrieved_docs]
    
    # Step 2: Rerank documents
    with torch.no_grad():
        rerank_results = rag_model.rerank([query], doc_contents, return_sorted=True)
    
    # Get top 5 reranked documents
    top_reranked = rerank_results[:5]
    
    # Build context
    context_parts = []
    for idx, score in top_reranked:
        context_parts.append(f"[Document {idx+1}]:\n{doc_contents[idx]}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Step 3: Generate answer
    try:
        client = OpenAI(
            api_key=st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        
        prompt = f"""You are a highly knowledgeable IT assistant. Answer the user's question using ONLY the information provided in the context below.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a comprehensive and detailed answer based on the context
- Include all relevant details, examples, and explanations from the documents
- If the context contains step-by-step instructions, include them
- If there are multiple relevant points, cover all of them
- Use specific information from the context (numbers, names, technical details)
- Organize your answer with paragraphs or bullet points if appropriate
- If the context doesn't contain enough information to fully answer the question, say so
- DO NOT make up information not in the context
- DO NOT say "according to the context" or "the document mentions" - just provide the information naturally

ANSWER:"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": "You are a helpful IT assistant who provides comprehensive, detailed answers based on provided documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        )
        
        answer = response.choices[0].message.content
        
        return answer, retrieved_docs, top_reranked
        
    except Exception as e:
        return f"Error generating response: {e}", retrieved_docs, top_reranked


def main():
    """Initialize and manage the IT Knowledge Base application interface."""
    st.set_page_config(
        page_title="IT Knowledge Base", 
        page_icon="🤖", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main header
    st.title("🤖 IT Knowledge Base")
    st.caption("Powered by NVIDIA RAG & AI Reranking")
    
    # Create three columns: left sidebar, main content, right sidebar (smaller)
    left_sidebar = st.sidebar
    
    # Main content area and right info panel (4:1 ratio instead of 3:1)
    main_col, right_col = st.columns([5, 1])
    
    # ============= LEFT SIDEBAR: Documents =============
    with left_sidebar:
        st.header("📚 Documents")
        
        # Upload section
        st.subheader("📤 Add Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload IT documentation in PDF format"
        )
        
        if uploaded_files:
            if st.button("Process & Add to Database", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success_count = 0
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    if add_to_db(uploaded_file):
                        success_count += 1
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                if success_count > 0:
                    st.success(f"✅ Added {success_count}/{len(uploaded_files)} documents!")
                    st.rerun()
                else:
                    st.error("❌ Failed to add documents")
        
        st.divider()
        
        # Document library
        st.subheader("📂 Document Library")
        
        if 'show_docs' not in st.session_state:
            st.session_state.show_docs = True
        
        col1, col2 = st.columns([3, 1])
        with col1:
            show_button = st.button(
                "Hide Documents" if st.session_state.show_docs else "Show Documents",
                use_container_width=True
            )
            if show_button:
                st.session_state.show_docs = not st.session_state.show_docs
        
        with col2:
            if st.button("🔄", help="Refresh"):
                st.rerun()
        
        if st.session_state.show_docs:
            with st.spinner("Loading documents..."):
                source_files = get_unique_source_files()
                
                if source_files:
                    st.info(f"**Total:** {len(source_files)} files")
                    
                    for source, chunk_count, date_added in source_files:
                        with st.expander(f"📄 {source}"):
                            st.write(f"**Chunks:** {chunk_count}")
                            st.write(f"**Added:** {date_added.strftime('%Y-%m-%d %H:%M')}")
                            
                            if st.button("Delete", key=f"del_{source}", type="secondary"):
                                with st.spinner(f"Deleting {source}..."):
                                    deleted = delete_documents_by_source(source)
                                    if deleted > 0:
                                        st.success(f"Deleted {deleted} chunks")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete")
                else:
                    st.warning("No documents in database")
    
    # ============= RIGHT SIDEBAR: System Info & Stats (COMPACT) =============
    with right_col:
        st.markdown("#### ℹ️ Info")
        
        # Database stats - compact
        doc_count = get_document_count()
        if doc_count >= 0:
            st.metric("Chunks", doc_count, label_visibility="visible")
        
        st.divider()
        
        # GPU/System info - compact
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            
            st.metric("GPU", f"{allocated:.1f}GB")
            st.progress(allocated / gpu_memory)
            st.caption(f"of {gpu_memory:.1f}GB")
        else:
            st.caption("CPU mode")
        
        st.divider()
        
        # API Configuration - compact
        with st.expander("⚙️ API"):
            api_key = st.text_input(
                "API Key",
                type="password",
                value=st.session_state.get("openai_api_key", ""),
                label_visibility="visible"
            )
            if api_key:
                st.session_state["openai_api_key"] = api_key
                st.caption("✅ Saved")
        
        st.divider()
        
        # Model info - very compact
        with st.expander("🤖 Models"):
            st.caption("**Embed:**")
            st.caption("llama-3.2-nv-embedqa")
            st.caption("**Rerank:**")
            st.caption("llama-3.2-nv-rerankqa")
            st.caption("**LLM:**")
            st.caption(os.getenv("OPENAI_MODEL", "gpt-4"))
    
    # ============= MAIN CONTENT: Query Interface =============
    with main_col:
        st.subheader("💬 Ask a Question")
        
        query = st.text_area(
            "Enter your question about IT and technology:",
            placeholder="e.g., What are the best practices for Docker containerization?",
            height=120,
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.button("🔍 Search & Answer", type="primary", use_container_width=True)
        
        if submit_button:
            if not query:
                st.warning("⚠️ Please enter a question")
            elif not st.session_state.get("openai_api_key") and not os.getenv("OPENAI_API_KEY"):
                st.error("🔑 Please configure your OpenAI API key in the right panel")
            else:
                # Show process in real-time with status updates
                run_rag_chain_streaming(query)
    # Sidebar Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("IT Knowledge Base RAG System")
    st.sidebar.caption("Powered by Oracle 23AI & NVIDIA RAG Models")                


def run_rag_chain_streaming(query):
    """Process a query using RAG with real-time status updates.
    
    Args:
        query (str): The user's question
    """
    # Create containers for each step
    step1_container = st.container()
    step2_container = st.container()
    step3_container = st.container()
    answer_container = st.container()
    
    # Step 1: Retrieve documents
    with step1_container:
        with st.status("🔍 Step 1: Retrieving similar documents...", expanded=True) as status:
            retrieved_docs = retrieve_similar_documents(query, top_k=10)
            
            if not retrieved_docs:
                st.error("No relevant documents found")
                status.update(label="❌ No documents found", state="error")
                return
            
            st.write(f"Retrieved {len(retrieved_docs)} documents")
            
            with st.expander("View retrieved documents"):
                for i, (content, metadata, similarity) in enumerate(retrieved_docs, 1):
                    st.write(f"**{i}.** Score: `{similarity:.4f}`")
                    st.caption(content[:200] + "...")
                    st.divider()
            
            status.update(label="✅ Step 1: Retrieved documents", state="complete")
    
    # Extract content for reranking
    doc_contents = [doc[0] for doc in retrieved_docs]
    
    # Step 2: Rerank documents
    with step2_container:
        with st.status("🎯 Step 2: Reranking documents...", expanded=True) as status:
            with torch.no_grad():
                rerank_results = rag_model.rerank([query], doc_contents, return_sorted=True)
            
            top_reranked = rerank_results[:5]
            st.write(f"Reranked to top {len(top_reranked)} most relevant")
            
            with st.expander("View reranked documents"):
                for i, (idx, score) in enumerate(top_reranked, 1):
                    st.write(f"**{i}.** Score: `{score:.4f}` (was position {idx+1})")
                    st.caption(doc_contents[idx][:200] + "...")
                    st.divider()
            
            status.update(label="✅ Step 2: Reranked documents", state="complete")
    
    # Build context
    context_parts = []
    for idx, score in top_reranked:
        context_parts.append(f"[Document {idx+1}]:\n{doc_contents[idx]}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Step 3: Generate answer
    with step3_container:
        with st.status("💭 Step 3: Generating answer...", expanded=True) as status:
            try:
                client = OpenAI(
                    api_key=st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                )
                
                prompt = f"""You are a highly knowledgeable IT assistant. Answer the user's question using ONLY the information provided in the context below.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Provide a comprehensive and detailed answer based on the context
- Include all relevant details, examples, and explanations from the documents
- If the context contains step-by-step instructions, include them
- If there are multiple relevant points, cover all of them
- Use specific information from the context (numbers, names, technical details)
- Organize your answer with paragraphs or bullet points if appropriate
- If the context doesn't contain enough information to fully answer the question, say so
- DO NOT make up information not in the context
- DO NOT say "according to the context" or "the document mentions" - just provide the information naturally

ANSWER:"""

                response = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4"),
                    messages=[
                        {"role": "system", "content": "You are a helpful IT assistant who provides comprehensive, detailed answers based on provided documentation. /no_think"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
                    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
                )
                
                answer = response.choices[0].message.content
                st.write("Answer generated successfully")
                status.update(label="✅ Step 3: Answer generated", state="complete")
                
            except Exception as e:
                st.error(f"Error: {e}")
                status.update(label="❌ Step 3: Error occurred", state="error")
                return
    
    # Display the final answer
    with answer_container:
        st.divider()
        display_answer_with_thinking(answer)


def display_answer_with_thinking(response_text):
    """Parse and display the response with thinking section shown separately.
    
    Args:
        response_text (str): The raw response from the LLM
    """
    # Check if there's a thinking section
    if "</think>" in response_text:
        # Split at the closing tag
        parts = response_text.split("</think>", 1)
        thinking_part = parts[0]
        answer_part = parts[1].strip() if len(parts) > 1 else ""
        
        # Remove opening <think> tag if present
        if "<think>" in thinking_part:
            thinking_part = thinking_part.split("<think>", 1)[1]
        
        # Display thinking in an info box
        st.info("💭 **Thinking Process**")
        st.write(thinking_part.strip())
        
        st.divider()
        
        # Display the actual answer
        if answer_part:
            st.success("📝 **Answer**")
            st.markdown(answer_part)
    else:
        # No thinking section, just display the answer
        st.success("📝 **Answer**")
        st.markdown(response_text)


             
if __name__ == "__main__":
    main()