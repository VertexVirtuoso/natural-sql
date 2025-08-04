"""Vector-based knowledge management for persistent learning."""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorKnowledgeManager:
    """Manages vector-based knowledge storage and retrieval for persistent learning."""
    
    def __init__(self, knowledge_dir: str = "./knowledge", vector_dir: str = "./vector_db"):
        """Initialize the vector knowledge manager.
        
        Args:
            knowledge_dir: Directory containing knowledge JSON files
            vector_dir: Directory for persistent vector storage
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.vector_dir = Path(vector_dir)
        self.settings = get_settings()
        
        # Initialize embeddings
        self._embeddings = self._setup_embeddings()
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=str(self.vector_dir),
            embedding_function=self._embeddings,
            collection_name="natural_sql_knowledge"
        )
        
        # Text splitter for long documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        logger.info(f"Vector knowledge manager initialized. Knowledge dir: {self.knowledge_dir}, Vector dir: {self.vector_dir}")
    
    def _setup_embeddings(self):
        """Set up embeddings using local sentence-transformers model."""
        try:
            # Use local sentence-transformers model instead of API-based embeddings
            # This avoids API costs and dependency on external services
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            
            # Use a good general-purpose embedding model
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.info("Local sentence-transformers embeddings initialized successfully")
            return embeddings
            
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to OpenAI embeddings")
            # Fallback to OpenAI embeddings if sentence-transformers not available
            try:
                if self.settings.openrouter_api_key:
                    # Try OpenRouter (may not support embeddings)
                    return OpenAIEmbeddings(
                        openai_api_base="https://openrouter.ai/api/v1",
                        openai_api_key=self.settings.openrouter_api_key,
                        model="text-embedding-ada-002"
                    )
                elif self.settings.openai_api_key:
                    # Use direct OpenAI
                    return OpenAIEmbeddings(
                        openai_api_key=self.settings.openai_api_key,
                        model="text-embedding-ada-002"
                    )
                else:
                    raise ValueError("Either OPENROUTER_API_KEY or OPENAI_API_KEY is required for embeddings")
            except Exception as e:
                logger.error(f"Failed to setup API-based embeddings: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Failed to setup embeddings: {e}")
            raise
    
    def is_first_run(self) -> bool:
        """Check if this is the first run (empty vector store)."""
        try:
            collection = self.vector_store.get()
            return len(collection.get('ids', [])) == 0
        except Exception as e:
            logger.warning(f"Could not check vector store status: {e}")
            return True
    
    def seed_initial_knowledge(self) -> None:
        """Load and vectorize initial knowledge from JSON files."""
        if not self.is_first_run():
            logger.info("Vector store already contains data, skipping initial seeding")
            return
        
        logger.info("First run detected, seeding initial knowledge...")
        
        try:
            # Load different types of knowledge
            self._load_domain_knowledge()
            self._load_query_patterns()
            self._load_business_rules()
            self._load_schema_documentation()
            
            logger.info("Initial knowledge seeding completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to seed initial knowledge: {e}")
            raise
    
    def _load_domain_knowledge(self) -> None:
        """Load K-pop domain knowledge from JSON file."""
        domain_file = self.knowledge_dir / "domain_knowledge.json"
        if not domain_file.exists():
            logger.warning(f"Domain knowledge file not found: {domain_file}")
            return
        
        try:
            with open(domain_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # Process terminology
            for term_data in data.get('kpop_terminology', []):
                doc_content = f"Terms: {', '.join(term_data['terms'])}\n"
                doc_content += f"Definition: {term_data['definition']}\n"
                doc_content += f"SQL Context: {term_data['sql_context']}"
                
                # Convert complex metadata to simple types
                metadata = {
                    "type": "terminology",
                    "terms": ', '.join(term_data['terms']),  # Convert list to string
                    "sql_context": term_data['sql_context'],
                    "source": "domain_knowledge"
                }
                
                documents.append(Document(
                    page_content=doc_content,
                    metadata=metadata
                ))
            
            # Process business rules
            for rule in data.get('business_rules', []):
                doc_content = f"Concept: {rule['concept']}\n"
                doc_content += f"Definition: {rule['definition']}\n"
                doc_content += f"SQL Logic: {rule['sql_logic']}"
                
                documents.append(Document(
                    page_content=doc_content,
                    metadata={
                        "type": "business_rule",
                        "concept": rule['concept'],
                        "sql_logic": rule['sql_logic'],
                        "source": "domain_knowledge"
                    }
                ))
            
            if documents:
                # Add documents to vector store (metadata is already simple)
                self.vector_store.add_documents(documents)
                logger.info(f"Loaded {len(documents)} domain knowledge documents")
            
        except Exception as e:
            logger.error(f"Failed to load domain knowledge: {e}")
    
    def _load_query_patterns(self) -> None:
        """Load successful query patterns from JSON file."""
        patterns_file = self.knowledge_dir / "query_patterns.json"
        if not patterns_file.exists():
            logger.warning(f"Query patterns file not found: {patterns_file}")
            return
        
        try:
            with open(patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # Process successful patterns
            for pattern in data.get('successful_patterns', []):
                doc_content = f"User Query: {pattern['user_query']}\n"
                doc_content += f"Successful SQL: {pattern['successful_sql']}\n"
                doc_content += f"Context: {pattern.get('context', '')}"
                
                documents.append(Document(
                    page_content=doc_content,
                    metadata={
                        "type": "successful_pattern",
                        "user_query": pattern['user_query'],
                        "sql": pattern['successful_sql'],
                        "context": pattern.get('context', ''),
                        "source": "query_patterns"
                    }
                ))
            
            # Process failed corrections
            for correction in data.get('failed_corrections', []):
                doc_content = f"Failed Query: {correction['failed_query']}\n"
                doc_content += f"Error: {correction['error']}\n"
                doc_content += f"Corrected Query: {correction['corrected_query']}\n"
                doc_content += f"Lesson: {correction['lesson']}"
                
                documents.append(Document(
                    page_content=doc_content,
                    metadata={
                        "type": "correction",
                        "failed_query": correction['failed_query'],
                        "corrected_query": correction['corrected_query'],
                        "lesson": correction['lesson'],
                        "source": "query_patterns"
                    }
                ))
            
            if documents:
                # Add documents to vector store (metadata is already simple)
                self.vector_store.add_documents(documents)
                logger.info(f"Loaded {len(documents)} query pattern documents")
            
        except Exception as e:
            logger.error(f"Failed to load query patterns: {e}")
    
    def _load_business_rules(self) -> None:
        """Load business rules from JSON file."""
        rules_file = self.knowledge_dir / "business_rules.json"
        if not rules_file.exists():
            logger.warning(f"Business rules file not found: {rules_file}")
            return
        
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            for rule in data.get('rules', []):
                doc_content = f"Rule: {rule['name']}\n"
                doc_content += f"Description: {rule['description']}\n"
                doc_content += f"SQL Implementation: {rule.get('sql_implementation', '')}\n"
                doc_content += f"Examples: {rule.get('examples', '')}"
                
                documents.append(Document(
                    page_content=doc_content,
                    metadata={
                        "type": "business_rule",
                        "rule_name": rule['name'],
                        "sql_implementation": rule.get('sql_implementation', ''),
                        "source": "business_rules"
                    }
                ))
            
            if documents:
                # Add documents to vector store (metadata is already simple)
                self.vector_store.add_documents(documents)
                logger.info(f"Loaded {len(documents)} business rule documents")
            
        except Exception as e:
            logger.error(f"Failed to load business rules: {e}")
    
    def _load_schema_documentation(self) -> None:
        """Load extended schema documentation from JSON file."""
        schema_file = self.knowledge_dir / "schema_documentation.json"
        if not schema_file.exists():
            logger.warning(f"Schema documentation file not found: {schema_file}")
            return
        
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            for table_doc in data.get('tables', []):
                doc_content = f"Table: {table_doc['name']}\n"
                doc_content += f"Purpose: {table_doc['purpose']}\n"
                doc_content += f"Key Fields: {', '.join(table_doc.get('key_fields', []))}\n"
                doc_content += f"Common Queries: {table_doc.get('common_queries', '')}\n"
                doc_content += f"Notes: {table_doc.get('notes', '')}"
                
                documents.append(Document(
                    page_content=doc_content,
                    metadata={
                        "type": "schema_doc",
                        "table_name": table_doc['name'],
                        "purpose": table_doc['purpose'],
                        "source": "schema_documentation"
                    }
                ))
            
            if documents:
                # Add documents to vector store (metadata is already simple)
                self.vector_store.add_documents(documents)
                logger.info(f"Loaded {len(documents)} schema documentation documents")
            
        except Exception as e:
            logger.error(f"Failed to load schema documentation: {e}")
    
    def find_similar_queries(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find similar queries from the knowledge base.
        
        Args:
            query: The user's natural language query
            k: Number of similar queries to return
            
        Returns:
            List of similar query documents with metadata
        """
        try:
            # Search for similar successful patterns
            similar_docs = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter={"type": {"$in": ["successful_pattern", "correction"]}}
            )
            
            results = []
            for doc, score in similar_docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                    "type": doc.metadata.get("type"),
                    "sql": doc.metadata.get("sql") or doc.metadata.get("corrected_query"),
                    "context": doc.metadata.get("context") or doc.metadata.get("lesson")
                })
            
            logger.info(f"Found {len(results)} similar queries for: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar queries: {e}")
            return []
    
    def find_domain_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Find relevant domain knowledge and business rules.
        
        Args:
            query: The user's natural language query
            k: Number of context documents to return
            
        Returns:
            List of relevant domain knowledge documents
        """
        try:
            # Search for domain knowledge and business rules
            context_docs = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter={"type": {"$in": ["terminology", "business_rule", "schema_doc"]}}
            )
            
            results = []
            for doc, score in context_docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                    "type": doc.metadata.get("type"),
                    "sql_context": doc.metadata.get("sql_context") or doc.metadata.get("sql_implementation")
                })
            
            logger.info(f"Found {len(results)} domain context items for: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find domain context: {e}")
            return []
    
    def learn_from_interaction(self, user_query: str, generated_sql: str, 
                             success: bool, error_msg: str = None, 
                             context: str = None) -> None:
        """Add new learning from user interaction to vector store.
        
        Args:
            user_query: The user's natural language query
            generated_sql: The SQL that was generated
            success: Whether the query was successful
            error_msg: Error message if query failed
            context: Additional context about the interaction
        """
        try:
            timestamp = datetime.now().isoformat()
            
            if success:
                # Add successful pattern
                doc_content = f"User Query: {user_query}\n"
                doc_content += f"Successful SQL: {generated_sql}\n"
                doc_content += f"Context: {context or 'User interaction'}\n"
                doc_content += f"Learned: {timestamp}"
                
                document = Document(
                    page_content=doc_content,
                    metadata={
                        "type": "successful_pattern",
                        "user_query": user_query,
                        "sql": generated_sql,
                        "context": context or "User interaction",
                        "source": "runtime_learning",
                        "timestamp": timestamp,
                        "learned_from": "interaction"
                    }
                )
            else:
                # Add failed attempt for future correction
                doc_content = f"Failed Query: {generated_sql}\n"
                doc_content += f"User Intent: {user_query}\n"
                doc_content += f"Error: {error_msg}\n"
                doc_content += f"Needs Correction: True\n"
                doc_content += f"Failed: {timestamp}"
                
                document = Document(
                    page_content=doc_content,
                    metadata={
                        "type": "failed_attempt",
                        "user_query": user_query,
                        "failed_sql": generated_sql,
                        "error": error_msg,
                        "source": "runtime_learning",
                        "timestamp": timestamp,
                        "needs_correction": True
                    }
                )
            
            self.vector_store.add_documents([document])
            
            # Also save to persistent JSON file for backup
            self._save_learning_to_file(user_query, generated_sql, success, error_msg, context, timestamp)
            
            logger.info(f"Learned from {'successful' if success else 'failed'} interaction: {user_query[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")
    
    def _save_learning_to_file(self, user_query: str, generated_sql: str, 
                              success: bool, error_msg: str, context: str, timestamp: str) -> None:
        """Save learning to persistent JSON file as backup."""
        try:
            learning_file = self.knowledge_dir / "runtime_learning.jsonl"
            
            # Create knowledge directory if it doesn't exist
            self.knowledge_dir.mkdir(parents=True, exist_ok=True)
            
            learning_entry = {
                "timestamp": timestamp,
                "user_query": user_query,
                "generated_sql": generated_sql,
                "success": success,
                "error_msg": error_msg,
                "context": context
            }
            
            # Append to JSONL file
            with open(learning_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(learning_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save learning to file: {e}")
    
    def add_correction(self, original_query: str, failed_sql: str, 
                      corrected_sql: str, lesson: str) -> None:
        """Add a correction for a previously failed query.
        
        Args:
            original_query: The original user query
            failed_sql: The SQL that failed
            corrected_sql: The corrected SQL
            lesson: What was learned from this correction
        """
        try:
            doc_content = f"Original Query: {original_query}\n"
            doc_content += f"Failed SQL: {failed_sql}\n"
            doc_content += f"Corrected SQL: {corrected_sql}\n"
            doc_content += f"Lesson: {lesson}"
            
            document = Document(
                page_content=doc_content,
                metadata={
                    "type": "correction",
                    "user_query": original_query,
                    "failed_query": failed_sql,
                    "corrected_query": corrected_sql,
                    "lesson": lesson,
                    "source": "manual_correction",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.vector_store.add_documents([document])
            logger.info(f"Added correction for query: {original_query[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to add correction: {e}")
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            collection = self.vector_store.get()
            ids = collection.get('ids', [])
            metadatas = collection.get('metadatas', [])
            
            total_docs = len(ids)
            
            # Count by type
            type_counts = {}
            source_counts = {}
            
            for metadata in metadatas:
                doc_type = metadata.get('type', 'unknown')
                source = metadata.get('source', 'unknown')
                
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
            
            return {
                "total_documents": total_docs,
                "types": type_counts,
                "sources": source_counts,
                "vector_store_path": str(self.vector_dir),
                "knowledge_path": str(self.knowledge_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {e}")
            return {"error": str(e)}
    
    def clear_knowledge_base(self) -> None:
        """Clear all knowledge from the vector store (for reset/debugging)."""
        try:
            # Delete the entire vector store directory
            import shutil
            if self.vector_dir.exists():
                shutil.rmtree(self.vector_dir)
            
            # Reinitialize empty vector store
            self.vector_store = Chroma(
                persist_directory=str(self.vector_dir),
                embedding_function=self._embeddings,
                collection_name="natural_sql_knowledge"
            )
            
            logger.info("Knowledge base cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear knowledge base: {e}")
            raise