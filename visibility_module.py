"""
TEAI Data Visibility Module
============================
Addresses the "I don't know my data" problem from your AI conversation.

Five visibility modes:
1. Corpus Profiler - What's in here?
2. Field Catalog - What are the important fields?
3. Workflow Reconstructor - What steps do people take?
4. Cross-Domain Linker - What enhances what?
5. Question Generator - What do people actually ask?
6. Schema Generator - How should this be in SQL?

This is NOT the Q&A interface - this is YOUR internal tool
for understanding and improving the knowledge base.
"""
from __future__ import annotations

import os
import json
import random
from typing import List, Dict, Any, Optional
from flask import Blueprint, jsonify, request

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Create Blueprint for visibility API routes
visibility_bp = Blueprint('visibility', __name__, url_prefix='/api/visibility')

# Will be set by main app
_vector_store = None
_llm = None


def init_visibility(vector_store, llm: ChatOpenAI):
    """Initialize the visibility module with vector store and LLM."""
    global _vector_store, _llm
    _vector_store = vector_store
    _llm = llm


def get_sample_chunks(n: int = 20, where_filter: Dict = None) -> List[str]:
    """Get random sample chunks from vector store."""
    if not _vector_store:
        return []
    
    try:
        # Use a generic query to get diverse results
        if where_filter:
            results = _vector_store.similarity_search(" ", k=n, filter=where_filter)
        else:
            results = _vector_store.similarity_search(" ", k=min(n, 50))
        
        chunks = [doc.page_content for doc in results]
        if len(chunks) > n:
            chunks = random.sample(chunks, n)
        return chunks
    except Exception as e:
        print(f"[!] Error getting sample chunks: {e}")
        return []


# ============================================================
# MODE 1: CORPUS PROFILER
# ============================================================

PROFILE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a data profiler analyzing a knowledge base about healthcare and trade certifications.

Analyze the sample chunks and provide a comprehensive profile:

1. **Document Types**: What kinds of documents/topics are represented?
2. **States Covered**: Which US states are mentioned?
3. **Certifications Found**: What certifications/credentials are discussed?
4. **Common Sections**: What section headings appear repeatedly?
5. **Key Entities**: Important organizations, agencies, programs mentioned
6. **Cost Information**: Range of costs mentioned
7. **Duration Information**: Training/certification timeframes mentioned
8. **Financial Aid Programs**: Aid programs mentioned
9. **Domain Vocabulary**: Specialized terms and acronyms
10. **Suggested Questions**: 15 realistic questions users would ask

Respond in JSON format."""),
    ("user", """Profile these {n_samples} sample chunks:

{chunks}""")
])


@visibility_bp.route('/profile', methods=['POST'])
def profile_corpus():
    """Mode 1: What's in this data?"""
    if not _llm:
        return jsonify({"error": "System not initialized"}), 503
    
    data = request.json or {}
    n_samples = data.get('n_samples', 25)
    category_filter = data.get('category', None)
    
    where_filter = None
    if category_filter:
        where_filter = {"certification": {"$eq": category_filter}}
    
    chunks = get_sample_chunks(n_samples, where_filter)
    
    if not chunks:
        return jsonify({"error": "No chunks found"}), 404
    
    chunks_text = "\n\n---\n\n".join(chunks)
    
    try:
        chain = PROFILE_PROMPT | _llm | JsonOutputParser()
        result = chain.invoke({
            "n_samples": len(chunks),
            "chunks": chunks_text
        })
        return jsonify({
            "status": "success",
            "samples_analyzed": len(chunks),
            "profile": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# MODE 2: FIELD CATALOG
# ============================================================

FIELD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a schema extractor analyzing healthcare certification content.

Extract ALL important fields/data points. For each field provide:
- field_name: Standardized name (e.g., "training_hours", "exam_fee")
- data_type: string, number, currency, duration, list, boolean
- examples: 2-3 actual values you found
- description: What this field represents
- required: Is this typically required information?
- categories: Which certifications use this field

Also identify:
- Relationships between fields
- Data that seems incomplete or missing
- Inconsistencies in how data is presented

Respond in JSON:
{{
    "fields": [
        {{
            "field_name": "...",
            "data_type": "...",
            "examples": ["..."],
            "description": "...",
            "required": true/false,
            "categories": ["..."]
        }}
    ],
    "relationships": ["field_a depends on field_b", ...],
    "missing_data": ["..."],
    "inconsistencies": ["..."]
}}"""),
    ("user", """Extract fields from this content about: {focus_area}

{chunks}""")
])


@visibility_bp.route('/fields', methods=['POST'])
def extract_fields():
    """Mode 2: What are the important fields?"""
    if not _llm:
        return jsonify({"error": "System not initialized"}), 503
    
    data = request.json or {}
    focus_area = data.get('focus', 'all healthcare certifications')
    n_samples = data.get('n_samples', 20)
    
    chunks = get_sample_chunks(n_samples)
    
    if not chunks:
        return jsonify({"error": "No chunks found"}), 404
    
    chunks_text = "\n\n---\n\n".join(chunks)
    
    try:
        chain = FIELD_PROMPT | _llm | JsonOutputParser()
        result = chain.invoke({
            "focus_area": focus_area,
            "chunks": chunks_text
        })
        return jsonify({
            "status": "success",
            "focus": focus_area,
            "catalog": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# MODE 3: WORKFLOW RECONSTRUCTOR
# ============================================================

WORKFLOW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are reconstructing a step-by-step certification process from documentation.

Create a COMPLETE workflow including:
1. Prerequisites before starting
2. Each step with requirements, duration, and cost
3. Common mistakes at each step
4. Tips for success
5. What happens after certification

Respond in JSON:
{{
    "process_name": "...",
    "overview": "...",
    "prerequisites": [
        {{"requirement": "...", "how_to_meet": "..."}}
    ],
    "total_estimated_cost": "...",
    "total_estimated_duration": "...",
    "steps": [
        {{
            "step_number": 1,
            "title": "...",
            "description": "...",
            "requirements": ["..."],
            "estimated_duration": "...",
            "estimated_cost": "...",
            "where_to_do_this": "...",
            "tips": ["..."],
            "common_mistakes": ["..."]
        }}
    ],
    "after_certification": {{
        "maintenance": "...",
        "renewal": "...",
        "career_paths": ["..."]
    }},
    "financial_aid_options": ["..."]
}}"""),
    ("user", """Reconstruct the complete workflow for: {process_name}

State: {state}

Content:
{chunks}""")
])


@visibility_bp.route('/workflow', methods=['POST'])
def reconstruct_workflow():
    """Mode 3: What steps do people take?"""
    if not _llm or not _vector_store:
        return jsonify({"error": "System not initialized"}), 503
    
    data = request.json or {}
    process_name = data.get('process', 'CNA Certification')
    state = data.get('state', 'Tennessee')
    
    # Search specifically for this process
    query = f"{process_name} {state} requirements steps process"
    results = _vector_store.similarity_search(query, k=12)
    chunks = [doc.page_content for doc in results]
    
    if not chunks:
        return jsonify({"error": "No content found for this process"}), 404
    
    chunks_text = "\n\n---\n\n".join(chunks)
    
    try:
        chain = WORKFLOW_PROMPT | _llm | JsonOutputParser()
        result = chain.invoke({
            "process_name": process_name,
            "state": state,
            "chunks": chunks_text
        })
        return jsonify({
            "status": "success",
            "process": process_name,
            "state": state,
            "workflow": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# MODE 4: CROSS-DOMAIN LINKER
# ============================================================

CROSSREF_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are finding connections between different certifications and programs.

Identify:
1. Shared requirements between certifications
2. Career progression paths (A leads to B leads to C)
3. Financial aid that covers multiple programs
4. Skills that transfer between certifications
5. Time/cost savings by combining certifications

Respond in JSON:
{{
    "career_paths": [
        {{
            "path_name": "...",
            "description": "...",
            "steps": [
                {{"certification": "...", "adds": "...", "time": "...", "cost": "..."}}
            ],
            "total_time": "...",
            "total_cost": "...",
            "ending_salary": "..."
        }}
    ],
    "shared_requirements": [
        {{
            "requirement": "...",
            "certifications": ["..."],
            "tip": "..."
        }}
    ],
    "financial_aid_matrix": [
        {{
            "program": "...",
            "covers": ["..."],
            "eligibility": "..."
        }}
    ],
    "transferable_skills": [
        {{
            "skill": "...",
            "from_certs": ["..."],
            "to_certs": ["..."]
        }}
    ],
    "efficiency_tips": ["..."]
}}"""),
    ("user", """Find cross-domain connections in this content:

{chunks}""")
])


@visibility_bp.route('/crossref', methods=['POST'])
def find_cross_references():
    """Mode 4: What enhances what?"""
    if not _llm:
        return jsonify({"error": "System not initialized"}), 503
    
    # Get diverse sample across categories
    chunks = get_sample_chunks(30)
    
    if not chunks:
        return jsonify({"error": "No chunks found"}), 404
    
    chunks_text = "\n\n---\n\n".join(chunks)
    
    try:
        chain = CROSSREF_PROMPT | _llm | JsonOutputParser()
        result = chain.invoke({"chunks": chunks_text})
        return jsonify({
            "status": "success",
            "connections": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# MODE 5: QUESTION GENERATOR
# ============================================================

QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are generating realistic user questions for a healthcare certification guidance system.

Generate questions that REAL users would ask. These should be specific, practical questions
that someone considering these certifications would actually type.

Categories:
1. **Getting Started** - "I know nothing, help me understand"
2. **Comparison** - "Which is better for my situation?"
3. **Requirements** - "What exactly do I need?"
4. **Cost/Affordability** - "Can I afford this?"
5. **Time/Schedule** - "How long and when?"
6. **Process** - "What are the exact steps?"
7. **Exam Prep** - "How do I pass?"
8. **After Certification** - "What happens next?"
9. **Edge Cases** - "What if I have special circumstances?"
10. **Financial Aid** - "How do I get help paying?"

Generate 5-7 specific questions per category based on the actual content.

Respond in JSON with category names as keys and arrays of questions as values."""),
    ("user", """Generate user questions based on:

Focus: {focus_area}

Content:
{chunks}""")
])


@visibility_bp.route('/questions', methods=['POST'])
def generate_questions():
    """Mode 5: What would users actually ask?"""
    if not _llm:
        return jsonify({"error": "System not initialized"}), 503
    
    data = request.json or {}
    focus_area = data.get('focus', 'all certifications')
    
    chunks = get_sample_chunks(20)
    
    if not chunks:
        return jsonify({"error": "No chunks found"}), 404
    
    chunks_text = "\n\n---\n\n".join(chunks)
    
    try:
        chain = QUESTION_PROMPT | _llm | JsonOutputParser()
        result = chain.invoke({
            "focus_area": focus_area,
            "chunks": chunks_text
        })
        return jsonify({
            "status": "success",
            "focus": focus_area,
            "questions": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# MODE 6: SQL SCHEMA GENERATOR
# ============================================================

SCHEMA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a database architect designing a SQL Server schema for healthcare certification data.

Based on the content, design a NORMALIZED schema that captures:
1. States and their specific requirements
2. Certifications and their details
3. Training programs and providers
4. Costs (which vary by state/program)
5. Financial aid programs
6. Career paths and progression

Provide complete T-SQL CREATE TABLE statements with:
- Appropriate data types (NVARCHAR, INT, DECIMAL(10,2), BIT, etc.)
- Primary keys and foreign keys
- Indexes for common queries
- Check constraints where appropriate
- Default values where sensible

Also provide:
- Sample INSERT statements (3-5 rows per table)
- Common query examples
- Notes on data that doesn't fit neatly

Respond in JSON:
{{
    "tables": [
        {{
            "name": "...",
            "description": "...",
            "create_sql": "CREATE TABLE ...",
            "sample_inserts": ["INSERT INTO ..."]
        }}
    ],
    "relationships": ["Table A.Column -> Table B.Column"],
    "common_queries": [
        {{"description": "...", "sql": "SELECT ..."}}
    ],
    "notes": ["..."]
}}"""),
    ("user", """Design a SQL Server schema for this content:

{chunks}""")
])


@visibility_bp.route('/schema', methods=['POST'])
def generate_schema():
    """Mode 6: SQL Schema Generator"""
    if not _llm:
        return jsonify({"error": "System not initialized"}), 503
    
    chunks = get_sample_chunks(25)
    
    if not chunks:
        return jsonify({"error": "No chunks found"}), 404
    
    chunks_text = "\n\n---\n\n".join(chunks)
    
    try:
        chain = SCHEMA_PROMPT | _llm | JsonOutputParser()
        result = chain.invoke({"chunks": chunks_text})
        return jsonify({
            "status": "success",
            "schema": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# SUMMARY ENDPOINT
# ============================================================

@visibility_bp.route('/summary', methods=['GET'])
def get_visibility_summary():
    """Get a quick summary of what visibility modes are available."""
    return jsonify({
        "modes": [
            {
                "name": "Corpus Profiler",
                "endpoint": "/api/visibility/profile",
                "method": "POST",
                "description": "What's in this data? Get an overview of the knowledge base.",
                "params": {"n_samples": "int (default 25)", "category": "string (optional)"}
            },
            {
                "name": "Field Catalog",
                "endpoint": "/api/visibility/fields",
                "method": "POST",
                "description": "What are the important fields? Extract structured data points.",
                "params": {"focus": "string (default 'all')", "n_samples": "int (default 20)"}
            },
            {
                "name": "Workflow Reconstructor",
                "endpoint": "/api/visibility/workflow",
                "method": "POST",
                "description": "What steps do people take? Reconstruct certification processes.",
                "params": {"process": "string (e.g., 'CNA Certification')", "state": "string"}
            },
            {
                "name": "Cross-Domain Linker",
                "endpoint": "/api/visibility/crossref",
                "method": "POST",
                "description": "What enhances what? Find connections between certifications."
            },
            {
                "name": "Question Generator",
                "endpoint": "/api/visibility/questions",
                "method": "POST",
                "description": "What would users ask? Generate realistic test questions.",
                "params": {"focus": "string (default 'all')"}
            },
            {
                "name": "Schema Generator",
                "endpoint": "/api/visibility/schema",
                "method": "POST",
                "description": "SQL Schema proposal based on content analysis."
            }
        ],
        "tip": "Start with /profile to understand your data, then use other modes to go deeper."
    })
