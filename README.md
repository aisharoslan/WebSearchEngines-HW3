# WSE HW3
Aisha Roslan (ar7805)

## Ranking Systems

### 1. BM25
Files:
* extract_subset.py
    - input: 8.8M full corpus in collection.tsv and 1M subset passage ids in msmarco_passages_subset.tsv
    - output: 1M subset corpus in subset_passages.tsv
* parsing.cpp
    - input: parses 1M subset corpus in subset_passages.tsv
    - output: 16 sorted temp files, page table with doc lengths for each passage id
* merging.cpp
    - input: 16 sorted temp files
    - output: 1 final sorted, merged postings file
* index.cpp
    - input: 1 merged, sorted postings file
    - output: metadata, lexicon, blocked and compressed inverted index
* querying.cpp
    - input: metadata, lexicon, blocked and compressed inverted index, page table, input queries, and qrels evaluation files
    - output: 6 files:
        - bm25.dev.top100.trec
        - bm25.dev.top1000.trec
        - bm25.eval.one.top100.trec
        - bm25.eval.one.top1000.trec
        - bm25.eval.two.top100.trec
        - bm25.eval.two.top1000.trec

### 2. HNSW
Files:
* hnsw.py
- ranks using hnsw index (dense retrieval) based on query/passage embeddings
- input: msmarco_passages_embeddings_subset.h5, msmarco_queries_dev_eval_embeddings.h5, qrels.dev.tsv, qrels.eval.one.tsv, qrels.eval.two.tsv
- output for (4,50,50): hnsw.low.dev.trec, hnsw.low.eval.one.trec, hnsw.low.eval.two.trec
- output for (8,200,200): hnsw.dev.trec, hnsw.eval.one.trec, hnsw.eval.two.trec

### 3. BM25 + Reranking
Files:
* rerank.py
- uses BM25 top 100/1000 candidate ids and reranks using dot product similarity
- input: msmarco_passages_embeddings_subset.h5, msmarco_queries_dev_eval_embeddings.h5, bm25.dev.top100.trec, bm25.eval.one.top100.trec, bm25.eval.two.top100.trec, bm25.dev.top1000.trec, bm25.eval.one.top1000.trec, bm25.eval.two.top1000.trec
- output (top 100): bm25_rerank_100.dev.trec, bm25_rerank_100.eval.one.trec, bm25_rerank_100.eval.two.trec
- output (top 1000): bm25_rerank_1000.dev.trec, bm25_rerank_1000.eval.one.trec, bm25_rerank_1000.eval.two.trec

### 4. BM25 + HNSW (Reciprocal Rank Fusion)
Files:
* fusion.py
- does rank fusion using Reciprocal Rank Fusion (RRF) for BM25 and HNSW top 100 ranked lists into 1 top 100 ranked list based on RRF score
- input: bm25 top 100, hnsw files
- output: fusion.dev.trec, fusion.eval.one.trec, fusion.eval.two.trec

## Documentation
- task.md: Explains HW3 task
- README.md: Explains files and ranking system implementations
- README.docx: Explains queries, qrels, and msmarco embeddings files
- read_h5.py: Code snippet for reading h5 files

