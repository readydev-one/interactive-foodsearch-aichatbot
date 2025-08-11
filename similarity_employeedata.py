#!/usr/bin/env python3
"""
Improved ChromaDB Similarity Search System
This module provides a robust employee search system using ChromaDB with proper error handling,
structured code organization, and comprehensive search capabilities.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmployeeSearchSystem:
    """
    A comprehensive employee search system using ChromaDB for similarity search
    and metadata filtering.
    """
    
    def __init__(self, collection_name: str = "employee_collection"):
        """Initialize the search system with ChromaDB client and embedding function."""
        self.collection_name = collection_name
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = None
        self.employees_data = self._get_employee_data()
        
    def _get_employee_data(self) -> List[Dict[str, Any]]:
        """Define comprehensive employee dataset."""
        return [
            {
                "id": "employee_1",
                "name": "John Doe",
                "experience": 5,
                "department": "Engineering",
                "role": "Software Engineer",
                "skills": "Python, JavaScript, React, Node.js, databases",
                "location": "New York",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_2",
                "name": "Jane Smith",
                "experience": 8,
                "department": "Marketing",
                "role": "Marketing Manager",
                "skills": "Digital marketing, SEO, content strategy, analytics, social media",
                "location": "Los Angeles",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_3",
                "name": "Alice Johnson",
                "experience": 3,
                "department": "HR",
                "role": "HR Coordinator",
                "skills": "Recruitment, employee relations, HR policies, training programs",
                "location": "Chicago",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_4",
                "name": "Michael Brown",
                "experience": 12,
                "department": "Engineering",
                "role": "Senior Software Engineer",
                "skills": "Java, Spring Boot, microservices, cloud architecture, DevOps",
                "location": "San Francisco",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_5",
                "name": "Emily Wilson",
                "experience": 2,
                "department": "Marketing",
                "role": "Marketing Assistant",
                "skills": "Content creation, email marketing, market research, social media management",
                "location": "Austin",
                "employment_type": "Part-time"
            },
            {
                "id": "employee_6",
                "name": "David Lee",
                "experience": 15,
                "department": "Engineering",
                "role": "Engineering Manager",
                "skills": "Team leadership, project management, software architecture, mentoring",
                "location": "Seattle",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_7",
                "name": "Sarah Clark",
                "experience": 8,
                "department": "HR",
                "role": "HR Manager",
                "skills": "Performance management, compensation planning, policy development, conflict resolution",
                "location": "Boston",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_8",
                "name": "Chris Evans",
                "experience": 20,
                "department": "Engineering",
                "role": "Senior Architect",
                "skills": "System design, distributed systems, cloud platforms, technical strategy",
                "location": "New York",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_9",
                "name": "Jessica Taylor",
                "experience": 4,
                "department": "Marketing",
                "role": "Marketing Specialist",
                "skills": "Brand management, advertising campaigns, customer analytics, creative strategy",
                "location": "Miami",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_10",
                "name": "Alex Rodriguez",
                "experience": 18,
                "department": "Engineering",
                "role": "Lead Software Engineer",
                "skills": "Full-stack development, React, Python, machine learning, data science",
                "location": "Denver",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_11",
                "name": "Hannah White",
                "experience": 6,
                "department": "HR",
                "role": "HR Business Partner",
                "skills": "Strategic HR, organizational development, change management, employee engagement",
                "location": "Portland",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_12",
                "name": "Kevin Martinez",
                "experience": 10,
                "department": "Engineering",
                "role": "DevOps Engineer",
                "skills": "Docker, Kubernetes, AWS, CI/CD pipelines, infrastructure automation",
                "location": "Phoenix",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_13",
                "name": "Rachel Brown",
                "experience": 7,
                "department": "Marketing",
                "role": "Marketing Director",
                "skills": "Strategic marketing, team leadership, budget management, campaign optimization",
                "location": "Atlanta",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_14",
                "name": "Matthew Garcia",
                "experience": 3,
                "department": "Engineering",
                "role": "Junior Software Engineer",
                "skills": "JavaScript, HTML/CSS, basic backend development, learning frameworks",
                "location": "Dallas",
                "employment_type": "Full-time"
            },
            {
                "id": "employee_15",
                "name": "Olivia Moore",
                "experience": 12,
                "department": "Engineering",
                "role": "Principal Engineer",
                "skills": "Technical leadership, system architecture, performance optimization, mentoring",
                "location": "San Francisco",
                "employment_type": "Full-time"
            },
        ]
    
    def _create_employee_documents(self) -> List[str]:
        """Create comprehensive text documents for each employee for similarity search."""
        documents = []
        for employee in self.employees_data:
            document = (
                f"{employee['role']} with {employee['experience']} years of experience "
                f"in {employee['department']}. Skills: {employee['skills']}. "
                f"Located in {employee['location']}. Employment type: {employee['employment_type']}."
            )
            documents.append(document)
        return documents
    
    def initialize_collection(self) -> bool:
        """Initialize and populate the ChromaDB collection."""
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass  # Collection doesn't exist, which is fine
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "A collection for storing employee data"}
            )
            logger.info(f"Created collection: {self.collection.name}")
            
            # Prepare data
            employee_documents = self._create_employee_documents()
            
            # Add data to collection
            self.collection.add(
                ids=[employee["id"] for employee in self.employees_data],
                documents=employee_documents,
                metadatas=[{
                    "name": employee["name"],
                    "department": employee["department"],
                    "role": employee["role"],
                    "experience": employee["experience"],
                    "location": employee["location"],
                    "employment_type": employee["employment_type"]
                } for employee in self.employees_data]
            )
            
            logger.info(f"Added {len(self.employees_data)} employees to collection")
            return True
            
        except Exception as error:
            logger.error(f"Error initializing collection: {error}")
            return False
    
    def similarity_search(self, query: str, n_results: int = 5, 
                         filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Perform similarity search with optional metadata filtering.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            Search results or None if error
        """
        if not self.collection:
            logger.error("Collection not initialized. Call initialize_collection() first.")
            return None
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters
            )
            return results
        except Exception as error:
            logger.error(f"Error in similarity search: {error}")
            return None
    
    def metadata_filter_search(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform metadata-only filtering without similarity search.
        
        Args:
            filters: Metadata filters to apply
            
        Returns:
            Filtered results or None if error
        """
        if not self.collection:
            logger.error("Collection not initialized. Call initialize_collection() first.")
            return None
            
        try:
            results = self.collection.get(where=filters)
            return results
        except Exception as error:
            logger.error(f"Error in metadata filtering: {error}")
            return None
    
    def get_collection_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the collection."""
        if not self.collection:
            logger.error("Collection not initialized.")
            return None
            
        try:
            all_items = self.collection.get()
            return {
                "total_documents": len(all_items['documents']),
                "collection_name": self.collection.name
            }
        except Exception as error:
            logger.error(f"Error getting collection stats: {error}")
            return None
    
    def print_search_results(self, results: Dict[str, Any], query: str = "", 
                           show_documents: bool = True, max_doc_length: int = 100):
        """
        Pretty print search results.
        
        Args:
            results: Search results from ChromaDB
            query: Original query (for display purposes)
            show_documents: Whether to show document content
            max_doc_length: Maximum length of document snippets to show
        """
        if not results or not results.get('ids'):
            print("No results found.")
            return
            
        if query:
            print(f"Query: '{query}'")
            
        # Handle both query results (with distances) and get results (without distances)
        has_distances = 'distances' in results and results['distances']
        ids_list = results['ids'][0] if isinstance(results['ids'][0], list) else results['ids']
        
        print(f"Found {len(ids_list)} results:")
        
        for i, doc_id in enumerate(ids_list):
            if has_distances:
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                document = results['documents'][0][i] if show_documents else ""
                print(f"\n  {i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
            else:
                metadata = results['metadatas'][i]
                document = results['documents'][i] if show_documents and 'documents' in results else ""
                print(f"\n  {i+1}. {metadata['name']} ({doc_id})")
            
            print(f"     Role: {metadata['role']}, Department: {metadata['department']}")
            print(f"     Experience: {metadata['experience']} years, Location: {metadata['location']}")
            
            if show_documents and document:
                doc_snippet = document[:max_doc_length] + "..." if len(document) > max_doc_length else document
                print(f"     Document: {doc_snippet}")


def run_comprehensive_demo():
    """Run a comprehensive demonstration of the search system."""
    # Initialize the search system
    search_system = EmployeeSearchSystem()
    
    if not search_system.initialize_collection():
        logger.error("Failed to initialize collection. Exiting.")
        return
    
    # Display collection statistics
    stats = search_system.get_collection_stats()
    if stats:
        print(f"\n=== Collection Statistics ===")
        print(f"Collection: {stats['collection_name']}")
        print(f"Total documents: {stats['total_documents']}")
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EMPLOYEE SEARCH DEMONSTRATION")
    print("="*60)
    
    # Demo 1: Similarity Search Examples
    print("\n=== SIMILARITY SEARCH EXAMPLES ===")
    
    search_queries = [
        ("Python developer with web development experience", 3),
        ("team leader manager with experience", 3),
        ("machine learning data science expert", 2),
        ("marketing strategy and analytics", 2)
    ]
    
    for i, (query, n_results) in enumerate(search_queries, 1):
        print(f"\n{i}. Searching for: {query}")
        print("-" * 50)
        results = search_system.similarity_search(query, n_results)
        if results:
            search_system.print_search_results(results, query)
    
    # Demo 2: Metadata Filtering Examples
    print("\n\n=== METADATA FILTERING EXAMPLES ===")
    
    filter_examples = [
        ({"department": "Engineering"}, "All Engineering employees"),
        ({"experience": {"$gte": 10}}, "Employees with 10+ years experience"),
        ({"location": {"$in": ["San Francisco", "Los Angeles"]}}, "Employees in California"),
        ({"employment_type": "Part-time"}, "Part-time employees")
    ]
    
    for i, (filters, description) in enumerate(filter_examples, 1):
        print(f"\n{i}. {description}:")
        print("-" * 50)
        results = search_system.metadata_filter_search(filters)
        if results:
            search_system.print_search_results(results, show_documents=False)
    
    # Demo 3: Combined Search (Similarity + Metadata Filtering)
    print("\n\n=== COMBINED SEARCH EXAMPLES ===")
    
    combined_searches = [
        (
            "senior Python developer full-stack",
            {"$and": [
                {"experience": {"$gte": 8}},
                {"location": {"$in": ["San Francisco", "New York", "Seattle"]}}
            ]},
            "Senior Python developers in major tech cities"
        ),
        (
            "leadership management experience",
            {"department": "Engineering"},
            "Engineering leaders and managers"
        )
    ]
    
    for i, (query, filters, description) in enumerate(combined_searches, 1):
        print(f"\n{i}. {description}:")
        print(f"   Query: '{query}' with metadata filters")
        print("-" * 50)
        results = search_system.similarity_search(query, n_results=5, filters=filters)
        if results:
            search_system.print_search_results(results, query)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED")
    print("="*60)


def main():
    """Main function to run the employee search system demonstration."""
    try:
        run_comprehensive_demo()
    except Exception as error:
        logger.error(f"Error in main execution: {error}")
        raise


if __name__ == "__main__":
    main()