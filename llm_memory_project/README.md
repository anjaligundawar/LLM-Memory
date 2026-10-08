# Memory Systems in LLMs: Review 2 code (Vector vs Graph under memory pollution)

## Files: who owns what
| File | Owner | What it does |
|---|---|---|
| `schema.py` | team | shared `MemoryUnit` (id, text, session, timestamp, tags, embedding) |
| `toy_data.py` | team | 10 clean memories (Priya and Rohan) plus test queries with correct answers |
| `embedder.py` | **Anjali** | text → vector (TF-IDF now; `SentenceEmbedder` ready for the final review) |
| `vector_store.py` | **Anjali** | builds the vector index, fills `embedding` |
| `vector_retriever.py` | **Anjali** | cosine similarity, top-k search |
| `entity_extractor.py` | **Prachita** | finds entities (Capitalised words) |
| `graph_store.py` | **Prachita** | builds the graph: node = entity, edge = mentioned together (with memory_id + text) |
| `graph_retriever.py` | **Prachita** | 2-hop BFS traversal search |
| `draw_graph.py` | **Prachita** | saves `graph.png` |
| `pollution_injector.py` | **Aksh** | adds 5 types of junk at any % and returns ground_truth |
| `memory_cleaner.py` | **Aksh** | hygiene v0: removes duplicates and stale facts |
| `metrics.py` | **Abhay** | Precision@K, Recall@K, MRR (minimal) |
| `demo_pipeline.py` | team | **vector vs graph side by side** → also saves `results.csv` + `graph.png` |
| `run_module3.py` | **Aksh** | before vs after cleaning |

## Run it (Windows)
1. Install Python from python.org and tick **"Add Python to PATH"**.
2. Open this folder in VS Code → Terminal → New Terminal.
3. Run:
   ```
   pip install -r requirements.txt
   python demo_pipeline.py
   python run_module3.py
   ```

## Next review
- **Anjali:** in `vector_retriever.py`, pass `SentenceEmbedder()` (`pip install sentence-transformers`),
  then swap the numpy search for FAISS (`pip install faiss-cpu`) or Chroma.
- **Prachita:** typed entities with spaCy (`en_core_web_sm`) or an LLM, typed relations
  (works_at, lives_in), and load the same nodes and edges into Neo4j (free Neo4j Desktop / AuraDB Free).
- **Aksh:** a real contradiction detector (NLI model or LLM), plus the full 0–75% × 5 types + mixed grid.
- **Abhay:** run the full grid × vector/graph × cleaning off/on × 3 seeds, then compute AUDC and recovery %.
- **Team:** `locomo_loader.py` turns each LoCoMo chat turn into a MemoryUnit; the QA "evidence" ids become the correct answers.
