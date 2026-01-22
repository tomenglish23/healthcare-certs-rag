"""
TEAI Agentic RAG System - Complete Redesign
============================================
A true multi-agent RAG architecture with:
- Query understanding and entity extraction
- Intelligent routing based on question type
- Multi-strategy retrieval
- Answer generation with grounding
- Self-critique and validation
- Confidence scoring based on evidence

Author: TEAI Healthcare Certifications Project
Version: 2.0.0
"""
from __future__ import annotations

import os
import sys
import re
import json
from typing import TypedDict, List, Dict, Any, Optional, Literal
from enum import Enum

from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict
import yaml

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# ============================================================
# CONFIGURATION
# ============================================================


def load_config() -> Dict[str, Any]:
    """Load config from YAML"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        print("[!] config.yaml not found, using defaults")
        return {
            'product': {'name': 'healthcare-certs', 'version': '2.0.0'},
            'branding': {'title': 'Healthcare Certifications', 'subtitle': 'Agentic RAG'},
            'data': {'source_file': 'healthcare-certs-all.md'},
            'features': {
                'show_confidence': True,
                'show_sources': True,
                'show_reasoning': True,
                'enable_self_critique': True
            },
            'taxonomies': {
                'states': [],
                'certifications': [],
                'cost_ranges': [],
                'durations': []
            }
        }


CONFIG = load_config()

PRODUCT_NAME = CONFIG['product']['name']
PRODUCT_VERSION = CONFIG['product']['version']
DATA_FILE = CONFIG['data']['source_file']

OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

print(f"[*] {PRODUCT_NAME} v{PRODUCT_VERSION} - Agentic RAG")
print(f"[*] Data: {DATA_FILE}")

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

# Global stores
vector_store = None
metadata_index = None  # For structured queries
app_graph = None

# ============================================================
# QUERY TYPES AND STATE
# ============================================================


class QueryType(str, Enum):
    """Types of queries we can handle"""
    COMPARISON = "comparison"  # "Compare CNA vs HHA in Tennessee"
    REQUIREMENTS = "requirements"  # "What are the requirements for CNA in TN?"
    COST_DURATION = "cost_duration"  # "How much does CNA cost? How long?"
    PROCESS = "process"  # "How do I become a CNA in Tennessee?"
    GENERAL = "general"  # General questions
    STUDY_MATERIAL = "study_material"  # "What should I study for the CNA exam?"
    RENEWAL = "renewal"  # "How do I renew my certification?"


class AgenticRAGState(TypedDict):
    """Complete state for the agentic workflow"""
    # Input
    question: str
    filters: Dict[str, str]  # From UI: state, certification, cost, duration
    
    # Query Understanding
    query_type: str
    extracted_entities: Dict[str, Any]  # state, cert_type, cost_preference, etc.
    search_queries: List[str]  # Reformulated queries for retrieval
    
    # Retrieval
    retrieved_docs: List[Document]
    retrieval_strategy: str
    
    # Generation
    draft_answer: str
    citations: List[Dict[str, str]]  # [{text: "...", source: "..."}]
    
    # Validation
    critique: str
    is_grounded: bool
    missing_info: List[str]
    
    # Final Output
    final_answer: str
    confidence: float
    reasoning_trace: List[str]  # For debugging/transparency
    sources: List[str]

# ============================================================
# DOCUMENT LOADING WITH RICH METADATA
# ============================================================


def load_documents() -> tuple[List[Document], Dict[str, Any]]:
    """
    Load markdown and extract both chunks and structured metadata.
    Returns (documents, metadata_index)
    """
    
    filepath = os.path.join("./data", DATA_FILE)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by headers for semantic chunking
    headers_to_split_on = [
        ("#", "state"),
        ("##", "certification"),
        ("###", "section"),
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    header_docs = header_splitter.split_text(content)
    
    # Further split large chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    all_docs = []
    metadata_index = {
        "states": set(),
        "certifications": set(),
        "state_certs": {},  # {state: [certs]}
        "cert_details": {}  # {(state, cert): {cost, duration, requirements}}
    }
    
    for doc in header_docs:
        # Extract metadata from headers
        state = doc.metadata.get("state", "").replace("# ", "").strip()
        cert = doc.metadata.get("certification", "").replace("## ", "").strip()
        section = doc.metadata.get("section", "").replace("### ", "").strip()
        
        # Build metadata index
        if state:
            metadata_index["states"].add(state)
            if state not in metadata_index["state_certs"]:
                metadata_index["state_certs"][state] = set()
        
        if cert and state:
            metadata_index["certifications"].add(cert)
            metadata_index["state_certs"][state].add(cert)
            
            # Extract structured data from content
            key = (state, cert)
            if key not in metadata_index["cert_details"]:
                metadata_index["cert_details"][key] = {}
            
            # Parse cost, duration from content
            text = doc.page_content.lower()
            
            # Cost extraction
            cost_match = re.search(r'\$[\d,]+(?:\s*-\s*\$[\d,]+)?', doc.page_content)
            if cost_match:
                metadata_index["cert_details"][key]["cost"] = cost_match.group()
            
            # Duration extraction
            duration_patterns = [
                r'(\d+)\s*(?:to\s*\d+\s*)?(?:weeks?|months?|hours?)',
                r'(\d+)-(\d+)\s*(?:weeks?|months?|hours?)'
            ]
            for pattern in duration_patterns:
                duration_match = re.search(pattern, text)
                if duration_match:
                    metadata_index["cert_details"][key]["duration"] = duration_match.group()
                    break
        
        # Enrich document metadata
        doc.metadata.update({
            "state": state,
            "certification": cert,
            "section": section,
            "source": DATA_FILE
        })
        
        # Split if too large
        if len(doc.page_content) > 1000:
            splits = text_splitter.split_documents([doc])
            all_docs.extend(splits)
        else:
            all_docs.append(doc)
    
    # Convert sets to lists for JSON serialization
    metadata_index["states"] = list(metadata_index["states"])
    metadata_index["certifications"] = list(metadata_index["certifications"])
    metadata_index["state_certs"] = {k: list(v) for k, v in metadata_index["state_certs"].items()}
    
    print(f"[*] Loaded {len(all_docs)} chunks")
    print(f"[*] Found {len(metadata_index['states'])} states, {len(metadata_index['certifications'])} cert types")
    
    return all_docs, metadata_index


def build_section_hierarchy(docs):
    """
    Build a nested structure:
    {
      "Tennessee": {
        "CNA": ["Overview", "Requirements", "Cost", "Duration", ...],
        "EMT": [...]
      },
      "West Virginia": { ... }
    }
    """
    hierarchy = defaultdict(lambda: defaultdict(list))

    for doc in docs:
        state = doc.metadata.get("state")
        cert = doc.metadata.get("certification")
        section = doc.metadata.get("section")

        if state and cert and section:
            clean_state = state.replace("#", "").strip()
            clean_cert = cert.replace("##", "").strip()
            clean_section = section.replace("###", "").strip()

            if clean_section not in hierarchy[clean_state][clean_cert]:
                hierarchy[clean_state][clean_cert].append(clean_section)

    return hierarchy


def create_vectorstore(docs: List[Document]) -> Chroma:
    """Create or load vectorstore with metadata filtering support"""
    
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    persist_dir = "./chroma_db_v2"
    
    # Always recreate for development - remove this in production
    if os.path.exists(persist_dir):
        print("[*] Loading existing vectorstore")
        vs = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        print(f"[*] Creating vectorstore with {len(docs)} documents")
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
    
    return vs

# ============================================================
# AGENT 1: QUERY ANALYZER
# ============================================================


def create_query_analyzer(llm: ChatOpenAI):
    """
    Analyzes the user's question to:
    1. Classify query type
    2. Extract entities (state, certification, cost preferences)
    3. Generate optimized search queries
    """
    
    analyzer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query analyzer for a healthcare certification information system.

Analyze the user's question and extract:
1. query_type: One of [comparison, requirements, cost_duration, process, general, study_material, renewal]
2. entities: Extract any mentioned states, certifications, cost preferences, duration preferences
3. search_queries: Generate 1-3 optimized search queries to find relevant information

Available states: {states}
Available certifications: {certifications}

Respond in JSON format:
{{
    "query_type": "...",
    "entities": {{
        "state": "extracted state or null",
        "certification": "extracted cert type or null",
        "cost_preference": "any cost constraints mentioned or null",
        "duration_preference": "any time constraints mentioned or null",
        "comparison_items": ["item1", "item2"] // if comparing
    }},
    "search_queries": ["query1", "query2"],
    "reasoning": "brief explanation of your analysis"
}}"""),
        ("user", "Question: {question}\nUI Filters: {filters}")
    ])
    
    def analyze(state: AgenticRAGState) -> AgenticRAGState:
        state["reasoning_trace"].append("🔍 Analyzing query...")
        
        try:
            chain = analyzer_prompt | llm | JsonOutputParser()
            
            result = chain.invoke({
                "question": state["question"],
                "filters": json.dumps(state["filters"]),
                "states": ", ".join(metadata_index.get("states", [])),
                "certifications": ", ".join(metadata_index.get("certifications", []))
            })
            
            state["query_type"] = result.get("query_type", "general")
            state["extracted_entities"] = result.get("entities", {})
            state["search_queries"] = result.get("search_queries", [state["question"]])
            
            # Merge UI filters with extracted entities (UI takes precedence)
            if state["filters"].get("state"):
                state["extracted_entities"]["state"] = state["filters"]["state"]
            if state["filters"].get("certification"):
                state["extracted_entities"]["certification"] = state["filters"]["certification"]
            
            state["reasoning_trace"].append(
                f"   Query type: {state['query_type']}, "
                f"Entities: {state['extracted_entities']}"
            )
            
        except Exception as e:
            print(f"[!] Query analysis error: {e}")
            state["query_type"] = "general"
            state["search_queries"] = [state["question"]]
            state["extracted_entities"] = {}
            state["reasoning_trace"].append(f"   ⚠️ Analysis fallback: {e}")
        
        return state
    
    return analyze

# ============================================================
# AGENT 2: SMART RETRIEVER
# ============================================================


def create_smart_retriever(vs: Chroma):
    """
    Multi-strategy retriever that adapts based on query type:
    - Uses metadata filtering when state/cert is known
    - Uses multiple queries for comparison questions
    - Adjusts k based on query complexity
    """
    
    def retrieve(state: AgenticRAGState) -> AgenticRAGState:
        state["reasoning_trace"].append("📚 Retrieving relevant documents...")
        
        query_type = state["query_type"]
        entities = state["extracted_entities"]
        search_queries = state["search_queries"]
        
        all_docs = []
        
        # Build metadata filter
        where_filter = None
        filter_conditions = []
        
        if entities.get("state"):
            filter_conditions.append({"state": {"$eq": entities["state"]}})
        if entities.get("certification"):
            filter_conditions.append({"certification": {"$eq": entities["certification"]}})
        
        if len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
        elif len(filter_conditions) > 1:
            where_filter = {"$and": filter_conditions}
        
        # Determine k based on query type
        k_values = {
            "comparison": 8,
            "requirements": 6,
            "cost_duration": 4,
            "process": 6,
            "study_material": 8,
            "renewal": 4,
            "general": 5
        }
        k = k_values.get(query_type, 5)
        
        # Execute searches
        for query in search_queries[:3]:  # Max 3 queries
            try:
                if where_filter:
                    docs = vs.similarity_search(
                        query,
                        k=k,
                        filter=where_filter
                    )
                else:
                    docs = vs.similarity_search(query, k=k)
                
                all_docs.extend(docs)
                
            except Exception as e:
                print(f"[!] Retrieval error for '{query}': {e}")
                # Fallback without filter
                docs = vs.similarity_search(query, k=k)
                all_docs.extend(docs)
        
        # Deduplicate while preserving order
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = hash(doc.page_content[:200])
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        
        state["retrieved_docs"] = unique_docs[:12]  # Cap at 12
        state["retrieval_strategy"] = f"filter={where_filter is not None}, k={k}, queries={len(search_queries)}"
        
        state["reasoning_trace"].append(
            f"   Retrieved {len(state['retrieved_docs'])} unique docs "
            f"(strategy: {state['retrieval_strategy']})"
        )
        
        return state
    
    return retrieve

# ============================================================
# AGENT 3: ANSWER GENERATOR WITH GROUNDING
# ============================================================


def create_answer_generator(llm: ChatOpenAI):
    """
    Generates answers that are grounded in retrieved context.
    Includes citation tracking and handles different query types.
    """
    
    # Different prompts for different query types
    prompts = {
        "comparison": """Compare the following items based on the provided context.
Create a clear comparison covering: requirements, cost, duration, and career outlook.
Use a structured format with clear sections for each item being compared.

Context:
{context}

Question: {question}

Provide a detailed comparison. For each fact, note which source it came from.""",

        "requirements": """Based on the provided context, list the specific requirements.
Be precise and include all details mentioned (hours, costs, prerequisites, etc.)

Context:
{context}

Question: {question}

List requirements clearly. Cite specific details from the sources.""",

        "cost_duration": """Extract and present cost and duration information from the context.
Include any ranges, variations, or conditions that affect the cost/duration.

Context:
{context}

Question: {question}

Provide specific numbers and any relevant conditions.""",

        "process": """Explain the step-by-step process based on the context.
Number each step and include relevant details like timelines and requirements.

Context:
{context}

Question: {question}

Provide a clear, actionable process.""",

        "study_material": """Based on the context, provide study guidance and exam information.
Include topics covered, recommended preparation, and exam format details.

Context:
{context}

Question: {question}

Provide comprehensive study guidance.""",

        "general": """Answer the question based ONLY on the provided context.
Be accurate and concise. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Provide a helpful, accurate answer."""
    }
    
    def generate(state: AgenticRAGState) -> AgenticRAGState:
        state["reasoning_trace"].append("✍️ Generating answer...")
        
        if not state["retrieved_docs"]:
            state["draft_answer"] = "I couldn't find relevant information to answer your question. Please try rephrasing or being more specific about the state or certification you're interested in."
            state["citations"] = []
            state["reasoning_trace"].append("   ⚠️ No documents retrieved")
            return state
        
        # Build context with source tracking
        context_parts = []
        sources_seen = set()
        
        for i, doc in enumerate(state["retrieved_docs"]):
            source_info = []
            if doc.metadata.get("state"):
                source_info.append(doc.metadata["state"])
            if doc.metadata.get("certification"):
                source_info.append(doc.metadata["certification"])
            if doc.metadata.get("section"):
                source_info.append(doc.metadata["section"])
            
            source_label = " > ".join(source_info) if source_info else f"Source {i+1}"
            sources_seen.add(source_label)
            
            context_parts.append(f"[{source_label}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Select prompt based on query type
        query_type = state["query_type"]
        prompt_template = prompts.get(query_type, prompts["general"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful healthcare certification advisor. "
                      "Answer based ONLY on the provided context. "
                      "Be accurate, specific, and cite your sources."),
            ("user", prompt_template)
        ])
        
        try:
            chain = prompt | llm
            response = chain.invoke({
                "context": context,
                "question": state["question"]
            })
            
            state["draft_answer"] = response.content
            state["citations"] = [{"source": s} for s in sources_seen]
            state["sources"] = list(sources_seen)
            
            state["reasoning_trace"].append(
                f"   Generated {len(state['draft_answer'])} char answer "
                f"with {len(state['sources'])} sources"
            )
            
        except Exception as e:
            print(f"[!] Generation error: {e}")
            state["draft_answer"] = "I encountered an error generating the answer. Please try again."
            state["reasoning_trace"].append(f"   ❌ Generation error: {e}")
        
        return state
    
    return generate

# ============================================================
# AGENT 4: SELF-CRITIQUE (OPTIONAL)
# ============================================================


def create_self_critique(llm: ChatOpenAI):
    """
    Validates the generated answer against the context.
    Checks for hallucinations and identifies missing information.
    """
    
    critique_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a fact-checker. Review the answer against the provided context.

Check for:
1. Accuracy: Is every claim in the answer supported by the context?
2. Completeness: Does the answer address the question fully?
3. Hallucination: Are there any facts not found in the context?

Respond in JSON:
{{
    "is_grounded": true/false,
    "issues": ["list of any issues found"],
    "missing_info": ["information that would improve the answer"],
    "confidence_adjustment": 0.0 to 1.0 (how confident we should be)
}}"""),
        ("user", """Context:
{context}

Question: {question}

Answer to verify:
{answer}

Evaluate this answer.""")
    ])
    
    def critique(state: AgenticRAGState) -> AgenticRAGState:
        # Skip if disabled or no answer
        if not CONFIG.get('features', {}).get('enable_self_critique', True):
            state["is_grounded"] = True
            state["critique"] = "Self-critique disabled"
            return state
        
        if not state["draft_answer"] or state["draft_answer"].startswith("I couldn't find"):
            state["is_grounded"] = False
            state["confidence"] = 0.1
            return state
        
        state["reasoning_trace"].append("🔎 Self-critique validation...")
        
        try:
            context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"][:5]])
            
            chain = critique_prompt | llm | JsonOutputParser()
            result = chain.invoke({
                "context": context,
                "question": state["question"],
                "answer": state["draft_answer"]
            })
            
            state["is_grounded"] = result.get("is_grounded", True)
            state["critique"] = "; ".join(result.get("issues", []))
            state["missing_info"] = result.get("missing_info", [])
            
            # Adjust confidence based on critique
            base_confidence = len(state["retrieved_docs"]) / 12  # Max docs = 12
            critique_factor = result.get("confidence_adjustment", 0.8)
            state["confidence"] = round(min(base_confidence * critique_factor, 1.0), 2)
            
            state["reasoning_trace"].append(
                f"   Grounded: {state['is_grounded']}, "
                f"Confidence: {state['confidence']}"
            )
            
        except Exception as e:
            print(f"[!] Critique error: {e}")
            state["is_grounded"] = True
            state["confidence"] = 0.5
            state["reasoning_trace"].append(f"   ⚠️ Critique fallback: {e}")
        
        return state
    
    return critique

# ============================================================
# AGENT 5: RESPONSE SYNTHESIZER
# ============================================================


def create_response_synthesizer(llm: ChatOpenAI):
    """
    Final step: Synthesizes the response, potentially regenerating
    if critique found issues, and formats for user consumption.
    """
    
    def synthesize(state: AgenticRAGState) -> AgenticRAGState:
        state["reasoning_trace"].append("📋 Synthesizing final response...")
        
        # If not grounded and we have context, try to regenerate
        if not state["is_grounded"] and state["retrieved_docs"] and state["confidence"] < 0.3:
            state["reasoning_trace"].append("   ⚠️ Low confidence, adding disclaimer")
            state["final_answer"] = (
                f"{state['draft_answer']}\n\n"
                f"⚠️ Note: This answer may be incomplete. "
                f"Please verify specific requirements with official sources."
            )
        else:
            state["final_answer"] = state["draft_answer"]
        
        # Add helpful context for certain query types
        if state["query_type"] == "process" and state["confidence"] > 0.5:
            state["final_answer"] += "\n\n💡 Tip: Requirements may change. Always verify with the state licensing board."
        
        state["reasoning_trace"].append(
            f"   Final confidence: {state['confidence']}, "
            f"Sources: {len(state['sources'])}"
        )
        
        return state
    
    return synthesize

# ============================================================
# BUILD THE AGENTIC GRAPH
# ============================================================


def create_agentic_graph(vs: Chroma) -> StateGraph:
    """
    Build the complete agentic RAG workflow.
    
    Flow:
    1. Query Analyzer → Understand question, extract entities
    2. Smart Retriever → Get relevant documents with filtering
    3. Answer Generator → Create grounded answer
    4. Self-Critique → Validate answer (optional)
    5. Response Synthesizer → Final formatting
    """
    
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    
    # Create all agents
    query_analyzer = create_query_analyzer(llm)
    smart_retriever = create_smart_retriever(vs)
    answer_generator = create_answer_generator(llm)
    self_critique = create_self_critique(llm)
    response_synthesizer = create_response_synthesizer(llm)
    
    # Build graph
    workflow = StateGraph(AgenticRAGState)
    
    # Add nodes
    workflow.add_node("analyze", query_analyzer)
    workflow.add_node("retrieve", smart_retriever)
    workflow.add_node("generate", answer_generator)
    workflow.add_node("critique", self_critique)
    workflow.add_node("synthesize", response_synthesizer)
    
    # Define edges (linear flow for now, can add conditionals later)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "critique")
    workflow.add_edge("critique", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()

# ============================================================
# API ROUTES
# ============================================================


@app.route('/')
def index():
    """Health check"""
    return jsonify({
        "status": "ok",
        "product": PRODUCT_NAME,
        "version": PRODUCT_VERSION,
        "architecture": "Agentic RAG"
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Return public config for frontend"""
    return jsonify({
        'product': CONFIG.get('product', {}),
        'branding': CONFIG.get('branding', {}),
        'features': CONFIG.get('features', {}),
        'sample_questions': CONFIG.get('sample_questions', {})
    })


@app.route('/api/taxonomies', methods=['GET', 'POST'])
def get_taxonomies():
    """Return taxonomies - now includes discovered metadata"""
    taxonomies = CONFIG.get('taxonomies', {}).copy()
    
    # Enhance with discovered metadata
    if metadata_index:
        if metadata_index.get("states"):
            taxonomies["discovered_states"] = metadata_index["states"]
        if metadata_index.get("certifications"):
            taxonomies["discovered_certifications"] = metadata_index["certifications"]
        if metadata_index.get("state_certs"):
            taxonomies["state_certifications"] = metadata_index["state_certs"]
    
    return jsonify(taxonomies)


@app.route('/api/query', methods=['POST'])
def query():
    """Handle search queries with full agentic pipeline"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        filters = data.get('filters', {})
        
        if not question:
            return jsonify({"error": "Question required"}), 400
        
        if not app_graph:
            return jsonify({"error": "System not initialized"}), 503
        
        # Initialize state
        initial_state = {
            "question": question,
            "filters": filters,
            "query_type": "general",
            "extracted_entities": {},
            "search_queries": [question],
            "retrieved_docs": [],
            "retrieval_strategy": "",
            "draft_answer": "",
            "citations": [],
            "critique": "",
            "is_grounded": True,
            "missing_info": [],
            "final_answer": "",
            "confidence": 0.0,
            "reasoning_trace": [],
            "sources": []
        }
        
        # Run the agentic pipeline
        result = app_graph.invoke(initial_state)
        
        # Build response
        response = {
            "answer": result["final_answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "query_type": result["query_type"],
            "entities": result["extracted_entities"]
        }
        
        # Include reasoning trace if enabled
        if CONFIG.get('features', {}).get('show_reasoning', False):
            response["reasoning"] = result["reasoning_trace"]
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/debug/metadata', methods=['GET'])
def debug_metadata():
    """Debug endpoint to see extracted metadata"""
    if metadata_index:
        return jsonify({
            "states": metadata_index.get("states", []),
            "certifications": metadata_index.get("certifications", []),
            "state_certs": metadata_index.get("state_certs", {}),
            "sample_details": dict(list(metadata_index.get("cert_details", {}).items())[:5])
        })
    return jsonify({"error": "Metadata not initialized"})


@app.route('/api/sections', methods=['GET'])
def get_sections():
    """Return the full L1/L2/L3 hierarchy for the Explorer UI."""
    return jsonify(section_hierarchy)

# This powers the left sidebar tree.


@app.route('/api/section-content', methods=['POST'])
def get_section_content():
    """
    Return the raw markdown content for a specific section.
    Input:
    {
      "state": "Tennessee",
      "certification": "CNA",
      "section": "Requirements"
    }
    """
    data = request.json
    state = data.get("state")
    cert = data.get("certification")
    section = data.get("section")

    if not (state and cert and section):
        return jsonify({"error": "Missing state/certification/section"}), 400

    matches = []
    for doc in docs:
        if (
            doc.metadata.get("state", "").strip("# ").strip() == state and
            doc.metadata.get("certification", "").strip("# ").strip() == cert and
            doc.metadata.get("section", "").strip("# ").strip() == section
        ):
            matches.append(doc.page_content)

    if not matches:
        return jsonify({"error": "Section not found"}), 404

    return jsonify({"content": "\n\n".join(matches)})

# This powers the main content panel.


@app.route('/api/section-metadata', methods=['POST'])
def get_section_metadata():
    """
    Return extracted metadata (cost, duration, requirements) for a section.
    """
    data = request.json
    state = data.get("state")
    cert = data.get("certification")

    if not (state and cert):
        return jsonify({"error": "Missing state/certification"}), 400

    key = (state, cert)
    details = metadata_index.get("cert_details", {}).get(key, {})

    return jsonify(details)

# This powers the Key Facts panel.


@app.route('/api/section-chunks', methods=['POST'])
def get_section_chunks():
    """
    Return all vectorstore chunks belonging to a section.
    """
    data = request.json
    state = data.get("state")
    cert = data.get("certification")
    section = data.get("section")

    if not (state and cert and section):
        return jsonify({"error": "Missing state/certification/section"}), 400

    results = []
    for doc in docs:
        if (
            doc.metadata.get("state", "").strip("# ").strip() == state and
            doc.metadata.get("certification", "").strip("# ").strip() == cert and
            doc.metadata.get("section", "").strip("# ").strip() == section
        ):
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

    return jsonify({"chunks": results})

# This powers the Chunk Viewer.


@app.route('/api/section-suggestions', methods=['POST'])
def get_section_suggestions():
    """
    Generate suggested questions for a section.
    """
    data = request.json
    state = data.get("state")
    cert = data.get("certification")
    section = data.get("section")

    if not (state and cert and section):
        return jsonify({"error": "Missing state/certification/section"}), 400

    # Build a prompt using the section content
    section_docs = []
    for doc in docs:
        if (
            doc.metadata.get("state", "").strip("# ").strip() == state and
            doc.metadata.get("certification", "").strip("# ").strip() == cert and
            doc.metadata.get("section", "").strip("# ").strip() == section
        ):
            section_docs.append(doc.page_content)

    context = "\n\n".join(section_docs)

    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate 10 helpful questions a user might ask after reading this section."),
        ("user", "{context}")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context})

    return jsonify({"suggestions": response.content.split("\n")})


study_memory = []

# This powers the Suggested Questions panel.


@app.route('/api/study/save', methods=['POST'])
def study_save():
    data = request.json
    study_memory.append(data)
    return jsonify({"status": "saved"})


@app.route('/api/study/list', methods=['GET'])
def study_list():
    return jsonify(study_memory)

# This powers the Learning Mode sidebar.

# ============================================================
# INITIALIZATION
# ============================================================


def initialize():
    """Initialize the agentic RAG system"""
    global vector_store, metadata_index, app_graph
    
    print("=" * 60)
    print("Initializing Agentic RAG System...")
    print("=" * 60)
    
    # Load documents and extract metadata
    docs, metadata_index = load_documents()
    section_hierarchy = build_section_hierarchy(docs)
    # Now your backend knows the full structure of the domain.
     
    # Create vector store
    vector_store = create_vectorstore(docs)
    
    # Build the agentic graph
    app_graph = create_agentic_graph(vector_store)
    
    # Initialize visibility module for data exploration
    try:
        from visibility_module import visibility_bp, init_visibility
        llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
        init_visibility(vector_store, llm)
        app.register_blueprint(visibility_bp)
        print("[*] Visibility module loaded - explore your data at /api/visibility/summary")
    except ImportError as e:
        print(f"[!] Visibility module not available: {e}")
    
    print("[*] Agentic RAG System ready!")
    print(f"[*] Agents: Query Analyzer → Smart Retriever → Answer Generator → Self-Critique → Synthesizer")
    return True

# ============================================================
# MAIN
# ============================================================


if __name__ == '__main__':
    if initialize():
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        print("[!] Initialization failed")
        sys.exit(1)
