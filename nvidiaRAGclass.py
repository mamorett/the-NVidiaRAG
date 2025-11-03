import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import List, Union, Tuple, Optional
import numpy as np


class RAGEmbeddingReranker:
    """
    A class for RAG (Retrieval-Augmented Generation) that handles both
    embedding generation and reranking using NVIDIA's models.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        reranking_model_name: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize the RAG embedding and reranking models.
        
        Args:
            embedding_model_name: HuggingFace model name for embeddings
            reranking_model_name: HuggingFace model name for reranking
            device: Device to run models on ('cuda:0', 'cpu', etc.). Auto-detects if None.
            max_length: Maximum sequence length for tokenization
        """
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        # Prefixes for embedding
        self.query_prefix = 'query: '
        self.document_prefix = 'passage: '
        
        # Initialize embedding model WITH BFLOAT16 to save memory
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(
            embedding_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16  # THIS IS THE KEY FIX
        )
        self.embedding_model = self.embedding_model.to(self.device)
        self.embedding_model.eval()
        
        # Initialize reranking model
        print(f"Loading reranking model: {reranking_model_name}")
        self.reranking_tokenizer = AutoTokenizer.from_pretrained(
            reranking_model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.reranking_tokenizer.pad_token is None:
            self.reranking_tokenizer.pad_token = self.reranking_tokenizer.eos_token
        
        self.reranking_model = AutoModelForSequenceClassification.from_pretrained(
            reranking_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        
        if self.reranking_model.config.pad_token_id is None:
            self.reranking_model.config.pad_token_id = self.reranking_tokenizer.eos_token_id
        
        self.reranking_model = self.reranking_model.to(self.device)
        
        print(f"Models loaded successfully on {self.device}")
        
        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU memory after loading models: {allocated:.2f} GB")
    
    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Average pooling with attention mask.
        
        Args:
            last_hidden_states: Hidden states from transformer model
            attention_mask: Attention mask for the input
            
        Returns:
            Normalized embeddings
        """
        last_hidden_states_masked = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embedding = F.normalize(embedding, dim=-1)
        return embedding
    
    def encode_queries(self, queries: List[str], batch_size: int = 8) -> torch.Tensor:
        """
        Encode queries into embeddings with mini-batching to avoid OOM.
        
        Args:
            queries: List of query strings
            batch_size: Number of queries to process at once
            
        Returns:
            Query embeddings tensor
        """
        all_embeddings = []
        
        # Process in mini-batches
        for i in range(0, len(queries), batch_size):
            batch_queries_list = queries[i:i + batch_size]
            
            # Add query prefix
            formatted_queries = [f"{self.query_prefix}{query}" for query in batch_queries_list]
            
            # Tokenize
            batch_queries = self.embedding_tokenizer(
                formatted_queries,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            batch_queries = {k: v.to(self.device) for k, v in batch_queries.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**batch_queries)
            
            # Apply average pooling
            embeddings = self.average_pool(
                outputs.last_hidden_state,
                batch_queries["attention_mask"]
            )
            
            # Keep in bfloat16, move to CPU to free GPU memory
            all_embeddings.append(embeddings.cpu())
            
            # Clean up GPU memory
            del batch_queries, outputs, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all embeddings and move back to device
        final_embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
        
        return final_embeddings
    
    def encode_documents(self, documents: List[str], batch_size: int = 8) -> torch.Tensor:
        """
        Encode documents into embeddings with mini-batching to avoid OOM.
        
        Args:
            documents: List of document strings
            batch_size: Number of documents to process at once
            
        Returns:
            Document embeddings tensor
        """
        all_embeddings = []
        
        # Process in mini-batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # Add document prefix
            formatted_documents = [f"{self.document_prefix}{doc}" for doc in batch_docs]
            
            # Tokenize
            batch_documents = self.embedding_tokenizer(
                formatted_documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            batch_documents = {k: v.to(self.device) for k, v in batch_documents.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**batch_documents)
            
            # Apply average pooling
            embeddings = self.average_pool(
                outputs.last_hidden_state,
                batch_documents["attention_mask"]
            )
            
            # Keep in bfloat16, move to CPU to free GPU memory
            all_embeddings.append(embeddings.cpu())
            
            # Clean up GPU memory
            del batch_documents, outputs, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all embeddings and move back to device
        final_embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
        
        return final_embeddings
    
    def compute_similarity(
        self,
        queries: Union[List[str], torch.Tensor],
        documents: Union[List[str], torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute similarity scores between queries and documents.
        
        Args:
            queries: List of query strings or pre-computed query embeddings
            documents: List of document strings or pre-computed document embeddings
            
        Returns:
            Similarity scores matrix (queries x documents)
        """
        # Encode if strings are provided
        if isinstance(queries, list):
            query_embeddings = self.encode_queries(queries)
        else:
            query_embeddings = queries
        
        if isinstance(documents, list):
            document_embeddings = self.encode_documents(documents)
        else:
            document_embeddings = documents
        
        # Compute similarity
        scores = query_embeddings @ document_embeddings.T
        return scores
    
    def rerank_prompt_template(self, query: str, passage: str) -> str:
        """
        Format query and passage with reranking prompt template.
        
        Args:
            query: Query string
            passage: Passage/document string
            
        Returns:
            Formatted prompt string
        """
        return f"question:{query} \n \n passage:{passage}"
    
    def rerank(
        self,
        queries: List[str],
        documents: List[str],
        return_sorted: bool = True,
        batch_size: int = 16
    ) -> Union[List[float], List[Tuple[int, float]]]:
        """
        Rerank documents for given queries with batched processing.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            return_sorted: If True, returns list of (doc_index, score) sorted by score.
                          If False, returns raw scores in order.
            batch_size: Batch size for reranking to avoid OOM
            
        Returns:
            Either list of scores or list of (index, score) tuples sorted by score
        """
        # Create pairs
        pairs = [[q, d] for q in queries for d in documents]
        
        # Apply prompt template
        texts = [self.rerank_prompt_template(query, doc) for query, doc in pairs]
        
        # Process in batches to avoid OOM
        all_scores = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            batch_dict = self.reranking_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            
            # Move to device
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            # Compute scores
            with torch.no_grad():
                logits = self.reranking_model(**batch_dict).logits
                batch_scores = logits.view(-1).cpu().tolist()
            
            all_scores.extend(batch_scores)
            
            # Clean up
            del batch_dict, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if return_sorted:
            # Return sorted (index, score) pairs per query
            num_docs = len(documents)
            results = []
            for q_idx in range(len(queries)):
                start_idx = q_idx * num_docs
                end_idx = start_idx + num_docs
                query_scores = all_scores[start_idx:end_idx]
                sorted_pairs = sorted(
                    enumerate(query_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                results.append(sorted_pairs)
            return results if len(queries) > 1 else results[0]
        
        return all_scores
    
    def retrieve_and_rerank(
        self,
        query: str,
        documents: List[str],
        top_k_retrieval: int = 10,
        top_k_rerank: int = 3
    ) -> List[Tuple[int, str, float]]:
        """
        Full RAG pipeline: retrieve top-k documents by embedding similarity,
        then rerank them for final results.
        
        Args:
            query: Query string
            documents: List of document strings
            top_k_retrieval: Number of documents to retrieve with embedding
            top_k_rerank: Number of documents to return after reranking
            
        Returns:
            List of (doc_index, document, rerank_score) tuples
        """
        # Step 1: Retrieve with embeddings
        similarity_scores = self.compute_similarity([query], documents)
        similarity_scores = similarity_scores[0].float().cpu().numpy()
        
        # Get top-k indices
        top_k_retrieval = min(top_k_retrieval, len(documents))
        top_indices = np.argsort(similarity_scores)[-top_k_retrieval:][::-1]
        
        # Get top documents
        top_documents = [documents[idx] for idx in top_indices]
        
        # Step 2: Rerank top documents
        rerank_results = self.rerank([query], top_documents, return_sorted=True)
        
        # Step 3: Return top-k reranked results
        final_results = []
        for local_idx, score in rerank_results[:top_k_rerank]:
            original_idx = top_indices[local_idx]
            final_results.append((original_idx, documents[original_idx], score))
        
        return final_results
    
    def batch_retrieve_and_rerank(
        self,
        queries: List[str],
        documents: List[str],
        top_k_retrieval: int = 10,
        top_k_rerank: int = 3
    ) -> List[List[Tuple[int, str, float]]]:
        """
        Batch version of retrieve_and_rerank for multiple queries.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            top_k_retrieval: Number of documents to retrieve with embedding
            top_k_rerank: Number of documents to return after reranking
            
        Returns:
            List of results for each query
        """
        results = []
        for query in queries:
            query_results = self.retrieve_and_rerank(
                query, documents, top_k_retrieval, top_k_rerank
            )
            results.append(query_results)
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    print("Initializing RAG Embedding and Reranker...")
    rag = RAGEmbeddingReranker()
    
    # Example queries and documents
    queries = [
        "how much protein should a female eat",
        "summit define",
    ]
    
    documents = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
        "Calorie intake should not fall below 1,200 a day in women or 1,500 a day in men, except under the supervision of a health professional."
    ]
    
    # Example 1: Compute embedding similarity
    print("\n" + "=" * 80)
    print("Example 1: Embedding Similarity")
    print("=" * 80)
    similarity_scores = rag.compute_similarity(queries, documents)
    print("\nSimilarity scores (queries x documents):")
    for i, query in enumerate(queries):
        print(f"\nQuery: '{query}'")
        for j, doc in enumerate(documents):
            print(f"  Doc {j}: {similarity_scores[i][j]:.4f} - {doc[:60]}...")
    
    # Example 2: Rerank documents
    print("\n" + "=" * 80)
    print("Example 2: Reranking")
    print("=" * 80)
    for i, query in enumerate(queries):
        print(f"\nQuery: '{query}'")
        rerank_results = rag.rerank([query], documents, return_sorted=True)
        print("Reranked results:")
        for rank, (doc_idx, score) in enumerate(rerank_results, 1):
            print(f"  {rank}. Doc {doc_idx} (Score: {score:.4f})")
            print(f"     {documents[doc_idx][:80]}...")
    
    # Example 3: Full RAG pipeline
    print("\n" + "=" * 80)
    print("Example 3: Full RAG Pipeline (Retrieve + Rerank)")
    print("=" * 80)
    query = "how much protein should a female eat?"
    results = rag.retrieve_and_rerank(
        query,
        documents,
        top_k_retrieval=3,
        top_k_rerank=2
    )
    print(f"\nQuery: '{query}'")
    print(f"\nTop {len(results)} results after retrieval and reranking:")
    for rank, (doc_idx, doc, score) in enumerate(results, 1):
        print(f"\n{rank}. Document {doc_idx} (Rerank Score: {score:.4f}):")
        print(f"   {doc[:120]}...")
    
    # Example 4: Batch processing
    print("\n" + "=" * 80)
    print("Example 4: Batch Retrieve and Rerank")
    print("=" * 80)
    batch_results = rag.batch_retrieve_and_rerank(
        queries,
        documents,
        top_k_retrieval=3,
        top_k_rerank=2
    )
    for i, (query, query_results) in enumerate(zip(queries, batch_results)):
        print(f"\nQuery {i+1}: '{query}'")
        print("Top results:")
        for rank, (doc_idx, doc, score) in enumerate(query_results, 1):
            print(f"  {rank}. Doc {doc_idx} (Score: {score:.4f}): {doc[:60]}...")
