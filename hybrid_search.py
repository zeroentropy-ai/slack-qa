#!/usr/bin/env python3
"""
Hybrid search implementation using Turbopuffer vector + BM25 search.
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio

# Import embedding functions
# from embed_qwen import embed_qwen
from embed_openai import embed_openai


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    text: str
    doc_id: str
    vector_score: float      # ANN similarity score
    bm25_score: float       # BM25 relevance score
    combined_score: float   # RRF combined score
    rank: int


class TurbopufferHybridSearcher:
    """Hybrid search using Turbopuffer vector + BM25."""
    
    def __init__(self, provider: str):
        """
        Initialize hybrid searcher.
        
        Args:
            provider: 'qwen' or 'openai' for embedding provider
        """
        self.provider = provider
        self.api_key = os.getenv("TURBOPUFFER_API_KEY")
        
        if not self.api_key:
            raise ValueError("TURBOPUFFER_API_KEY environment variable not set")
        
        # Import turbopuffer
        try:
            import turbopuffer as tpuf
            self.tpuf = tpuf
        except ImportError:
            raise ImportError("Install turbopuffer: pip install turbopuffer")
        
        print(f"Initialized Turbopuffer hybrid searcher with {provider} embeddings")
    
    def _get_namespace(self, tenant_name: str):
        """Get Turbopuffer namespace for tenant and provider."""
        namespace_name = f"{tenant_name}"
        tpuf_client = self.tpuf.Turbopuffer(api_key=self.api_key, region='aws-us-west-2')
        return tpuf_client.namespace(namespace_name)
    
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate query embedding based on provider."""
        if self.provider == 'qwen':
            embedding = await embed_qwen(query, "query")
        elif self.provider == 'openai':
            embedding = await embed_openai(query)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        return embedding
    
    def _reciprocal_rank_fusion(self, 
                               vector_results: List[Dict], 
                               bm25_results: List[Dict], 
                               k: int = 60) -> List[Tuple[Dict, float]]:
        """
        Combine vector and BM25 results using Reciprocal Rank Fusion.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: RRF parameter (default 60)
            
        Returns:
            List of (document, combined_score) tuples
        """
        # Create score mappings
        vector_scores = {}
        bm25_scores = {}
        all_docs = {}
        
        # Process vector results
        vector_docs = vector_results.rows if hasattr(vector_results, 'rows') else []
        for rank, result in enumerate(vector_docs):
            doc_id = result.id
            vector_scores[doc_id] = 1.0 / (rank + k)
            all_docs[doc_id] = result
        
        # Process BM25 results  
        bm25_docs = bm25_results.rows if hasattr(bm25_results, 'rows') else []
        for rank, result in enumerate(bm25_docs):
            doc_id = result.id
            bm25_scores[doc_id] = 1.0 / (rank + k)
            all_docs[doc_id] = result
        
        # Combine scores using RRF
        combined_results = []
        for doc_id, doc in all_docs.items():
            vector_score = vector_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)
            combined_score = vector_score + bm25_score
            
            combined_results.append((doc, combined_score))
        
        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    async def search(self, 
                    tenant_name: str, 
                    query: str, 
                    top_k: int = 20) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining vector and BM25 results.
        
        Args:
            tenant_name: Name of tenant to search
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of HybridSearchResult objects
        """
        # Get namespace
        ns = self._get_namespace(tenant_name)
        
        # Generate query embedding
        query_embedding = await self._generate_query_embedding(query)
        
        # Perform multi-query search
        try:
            # Truncate query for BM25
            if len(query) > 1024:
                query = query[:1024]

            result = ns.multi_query(queries=[
                # Vector search (ANN)
                {'rank_by': ['vector', 'ANN', query_embedding], 'top_k': top_k * 2, 'include_attributes': True},
                # Text search (BM25)
                {'rank_by': ['content', 'BM25', query], 'top_k': top_k * 2, 'include_attributes': True}
            ])
            
            # Extract results from response object
            if not result or not hasattr(result, 'results'):
                print(f"Warning: multi_query returned unexpected response: {type(result)}")
                return []
            
            if len(result.results) < 2:
                print(f"Warning: multi_query returned {len(result.results)} result sets, expected 2")
                return []
                
            vector_results = result.results[0] if len(result.results) > 0 else []
            bm25_results = result.results[1] if len(result.results) > 1 else []
            
        except Exception as e:
            print(f"Error in multi_query search: {e}")
            return []
        
        # Apply RRF fusion
        combined_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
        
        # Get docs for later reference
        vector_docs = vector_results.rows if hasattr(vector_results, 'rows') else []
        bm25_docs = bm25_results.rows if hasattr(bm25_results, 'rows') else []
        
        # Convert to HybridSearchResult objects
        search_results = []
        for rank, (doc, combined_score) in enumerate(combined_results[:top_k]):
            # Get individual scores (0 if not present in respective result set)  
            vector_score = 0.0
            for i, v_result in enumerate(vector_docs):
                if v_result.id == doc.id:
                    vector_score = 1.0 / (i + 1)  # Simple rank-based score
                    break
            
            bm25_score = 0.0
            for i, b_result in enumerate(bm25_docs):
                if b_result.id == doc.id:
                    bm25_score = 1.0 / (i + 1)  # Simple rank-based score
                    break
            
            # Extract text, doc_id from doc
            result = HybridSearchResult(
                text=doc.content,
                doc_id=doc.id,
                vector_score=vector_score,
                bm25_score=bm25_score,
                combined_score=combined_score,
                rank=rank + 1
            )
            search_results.append(result)
        
        return search_results
    
    async def search_file(self, 
                         tenant_name: str, 
                         query: str, 
                         top_k: int = 20) -> List[Dict]:
        """
        Compatibility method that returns results in the same format as ChunkSearcher.
        
        Returns:
            List of dictionaries with 'text', 'url', 'chunk_idx', 'similarity' keys
        """

        print(f"searching {tenant_name} for '{query}' (top_k={top_k})")
        results = await self.search(tenant_name, query, top_k)
        
        # Convert to ChunkSearcher format
        compat_results = []
        for result in results:
            compat_result = {
                'text': result.text,
                'doc_id': result.doc_id,
                'similarity': result.combined_score  # Use combined score as similarity
            }
            compat_results.append(compat_result)
        
        return compat_results


async def test_hybrid_search():
    """Test the hybrid search functionality."""
    
    # Test with a simple query
    searcher = TurbopufferHybridSearcher('openai')
    
    query = "How to create a user account"
    tenant = "bltsmrt_notion_site_documents"
    
    print(f"Testing hybrid search: '{query}' on {tenant}")
    
    try:
        results = await searcher.search(tenant, query, top_k=5)
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Score: {result.combined_score:.4f} (v:{result.vector_score:.3f}, b:{result.bm25_score:.3f})")
            print(f"   URL: {result.url}")
            print(f"   Chunk: {result.chunk_idx}")
            print(f"   Text: {result.text[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_hybrid_search())