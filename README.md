# TEAI Agentic RAG System v2.0

## The Problem We Solved

Your original system was "RAG without the AG" - it used embeddings 
for retrieval but didn't leverage LLM intelligence for:
- Understanding what the user was actually asking
- Routing to the right retrieval strategy
- Validating that answers were grounded in evidence
- Handling different question types appropriately

## New Architecture: True Agentic RAG

+--------------------------------------------------+
|          USER QUERY + FILTERS                    |
+--------|-----------------------------------------+
+--------v-----------------------------------------+
| AGENT 1: QUERY ANALYZER                          |
| Classifies QUERY TYPE  (comparison, reqs, proc)  |
| Extracts ENTITIES      (state, cert, cost prefs) |
| Generates optimized SEARCH QUERIES               |
| Merges with UI FILTER SELECTIONS                 |
+--------|-----------------------------------------+
+--------v-----------------------------------------+
| AGENT 2: SMART RETRIEVER                         |
| Builds METADATA FILTERS from Extracted ENTITIES  |
| Adjusts k based on QUERY TYPE    (k = # of docs) |
| Executes multiple SEARCH QUERIES for comparisons |
| Deduplicates and ranks results                   |
+--------|-----------------------------------------+
+--------v-----------------------------------------+
| AGENT 3: ANSWER GENERATOR                        |
| By QUERY TYPE: Selects:   PROMPT TEMPLATE        |
| From CONTEXT,  Generates: grounded ANSWER        |
|                Tracks:    CITATIONS & SOURCES    |
| Appropriately, Formats:   RESPONSE               |
+--------|-----------------------------------------+
+--------v-----------------------------------------+
| AGENT 4: SELF-CRITIQUE                           |
| Validates ANSWER against CONTEXT                 |
| Checks for hallucinations                        |
| Identifies missing information                   |
| Adjusts CONFIDENCE SCORE                         |
+--------|-----------------------------------------+
+--------v-----------------------------------------+
| AGENT 5: RESPONSE SYNTHESIZER                    |
| If low CONFIDENCE: Adds DISCLAIMERS              |
| Formats final RESPONSE                           |
| By QUERY TYPE: Adds: helpful TIPS                |
+--------|-----------------------------------------+
+--------v-----------------------------------------+
| FINAL RESPONSE                                   |
| Answer           (grounded in evidence)          |
| Confidence score (based on retrieval + critique) |
| Sources          (tracked through pipeline)      |
| Query type       (for UI display)                |
| Reasoning trace  (for transparency)              |
+--------------------------------------------------+

## Query Types

The system recognizes and handles different question types:

| Type           | Example                      | Handling                               |
|----------------|------------------------------|----------------------------------------|
| comparison     | "Compare CNA vs EMT"         | Retrieves both, structured comparison  |
| requirements   | "What are CNA requirements?" | Focuses on prerequisites, hours, costs |
| cost_duration  | "How much does CNA cost?"    | Extracts specific numbers              |
| process        | "How do I become a CNA?"     | Step-by-step format                    |
| study_material | "What's on the CNA exam?"    | Exam content and prep tips             |
| renewal        | "How do I renew my CNA?"     | Continuing education focus             |
| general        | "Tell me about healthcare"   | Broad overview                         |

## Key Improvements Over v1.0

### 1. Filters Actually Work
```python
# OLD: Filters were UI-only, ignored in retrieval
docs = vs.similarity_search(question, k=4)

# NEW: Filters become metadata queries
where_filter = {"$and": [
    {"state": {"$eq": "Tennessee"}},
    {"certification": {"$eq": "CNA"}}
]}
docs = vs.similarity_search(question, k=k, filter=where_filter)

### 2. Query Understanding
```python
# OLD: Raw question goes straight to embedding
embedding = embed(question)

# NEW: LLM analyzes the question first
{
    "query_type": "comparison",
    "entities": {"state": "Tennessee", "certification": "CNA"},
    "search_queries": ["CNA requirements Tennessee", "CNA cost TN"]
}

### 3. Adaptive Retrieval
```python
# OLD: Always k=4
docs = vs.similarity_search(question, k=4)

# NEW: k varies by query type
k_values = {
    "comparison": 8,    # Need more docs to compare
    "requirements": 6,  # Comprehensive coverage
    "cost_duration": 4, # Focused retrieval
    "general": 5        # Balanced
}

### 4. Self-Critique
```python
# OLD: No validation
answer = llm.invoke(prompt)
return answer

# NEW: Validate before returning
critique = validate_against_context(answer, context)
if not critique.is_grounded:
    confidence *= 0.5
    add_disclaimer(answer)

### 5. Type-Specific Prompts
```python
# OLD: One prompt for everything
prompt = "Answer based on context..."

# NEW: Specialized prompts
prompts = {
    "comparison": "Compare the following items...",
    "requirements": "List the specific requirements...",
    "process": "Explain the step-by-step process...",
}

## Visibility Module (Data Exploration)

The visibility module addresses the "I don't know my data" problem with 6 modes:

| Mode             | Endpoint                  | Purpose                        |
|------------------|---------------------------|--------------------------------|
| Corpus Profiler  | /api/visibility/profile   | What's in this data?           |
| Field Catalog    | /api/visibility/fields    | What are the important fields? |
| Workflow Mapper  | /api/visibility/workflow  | What steps do people take?     |
| Cross-Ref Linker | /api/visibility/crossref  | What enhances what?            |
| Question Gen     | /api/visibility/questions | What would users ask?          |
| Schema Generator | /api/visibility/schema    | How should this be in SQL?     |

## File Structure

agentic_rag/
|-- app.py                       # Main backend with all agents
|-- visibility_module.py         # Data exploration tools
|-- config.yaml                  # Configuration and taxonomies
|-- TEAIAgenticRAG.jsx           # React frontend component
|-- requirements.txt             # Python dependencies
|-- data/
|   +-- healthcare-certs-all.md  # Your knowledge base
+-- chroma_db_v2/                # Vector store (auto-created)

## Deployment

### Backend (Render)

requirements.txt includes:
- flask, flask-cors
- langchain, langchain-openai, langchain-chroma
- langgraph
- chromadb
- pyyaml, openai

### Environment Variables

OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
PORT=5000

### Frontend Integration

```jsx
import TEAIAgenticRAG from './components/TEAIAgenticRAG';

// Set API base URL via env:
// REACT_APP_API_BASE=https://your-api.onrender.com

## Confidence Scoring

Confidence is calculated based on:

1. Retrieval Quality (0-40%): How many relevant docs were found
2. Entity Match      (0-30%): Did we find content matching the filters
3. Self-Critique     (0-30%): Did the answer pass validation

Final Confidence = (retrieval * 0.4) + (entity_match * 0.3) + (critique * 0.3)

## Reasoning Trace

When enabled, the system provides a trace of its reasoning:

[Analyzing query...]             Query type: requirements, Entities: {state: TN, cert: CNA}
[Retrieving relevant docs...]    Rxed 6 unique docs (strategy: filter=true, k=6, queries=2)
[Generating answer...]           Generated 847 char answer with 3 sources
[Self-critique validation...]    Grounded: True, Confidence:  0.78
[Synthesizing final response...] Final confidence:            0.78, Sources: 3

## Future Enhancements

1. Conversation Memory: Multi-turn dialogue support
2. User Feedback Loop: Learn from thumbs up/down
3. Document Upload: Add custom knowledge sources
4. Comparison Tables: Auto-generate comparison matrices
5. Career Path Visualization: Interactive progression charts

---

This is now a TRUE Agentic RAG system - the LLM is involved at every step,
not just for generating the final answer. Each agent has a specific role,
and together they provide more accurate, well-grounded responses.
#   h e a l t h c a r e - c e r t s - r a g  
 