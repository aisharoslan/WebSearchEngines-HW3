HW03

Task
- Compare BM25 with Dense Vector Retrieval
- Use 3 search systems
    1. BM25 index
    2. Graph-based vector search index (HNSW)
        - in-memory index
        - use 1M passages from MS-MARCO (h5 format)
            - python snippets to read h5 file
            - h5 file has ids of passages/queries + corresponding 384-dim embeddings
        - tsv file with all passage ids
        - h5 file with embeddings of all queries that need to be evaluated
            - embeddings of all queries in queries.eval.tsv and queries.dev.tsv
            - embeddings generated using dot product as similarity measure -> what you use to build vector index
        
    3. Re-ranking system
        - generate list of candidate docs using BM25 (top 100 or 1000)
        - re-rank by computing similarity of vector embeddings

DVR vs BM25
- DVR captures semantic similarity when query terms/relevant docs don't share same words
- DVR encodes both queries and docs as high-dimensional vectors using pre-trained language models (BERT/MiniLM)
- Query and Doc relevance determined by similarity between vector repr (cosine similarity)

Steps
1. Rebuild BM25 index using subset passages
2. Build vector index
    - use faiss library (has various ANN vector search algo)
    - if use hnsw, careful with parameter selection
        - m
        - ef_construction
        - ef_search
        - (4,50,50) to (8, 200, 200)
        - higher values -> higher accuracy but increased search time
3. Evaluation - 3 files
    1. qrel.eval.one.tsv - query_id, doc_id, relevance_label (0-3)
    2. qrel.eval.two.tsv - query_id, doc_id, relevance_label (0-3)
    3. qrels.dev.tsv - query_id, doc_id, relevance (0/1)

    Actual Queries for query_ids in these files are in:
        - queries.eval.tsv
        - queries.dev.tsv

4. Run queries on all 3 systems and evaluate MRR@10, Recall@100, NDCG@10, NDCG@100
    - use MAP instead of NDCG for queries in qrels.dev.tsv bc NDCG only for multiple levels of relevance
    - use trec_eval github to compute these measures

5. Submission
    - submit code and report with analysis
    - describe tradeoffs in terms of retrieval quality and overall efficiency (time and memory)
    - read posted research papers on reading list
    - python - most convenient