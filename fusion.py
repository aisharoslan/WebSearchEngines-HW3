from collections import defaultdict

K = 100
RRF_K = 60 # constant added to ranks in RRF formula


def read_trec_run(filename):
    """
    Reads TREC run file and returns a dictionary 
        - query_id -> list of (doc_id, score)
    """
    runs = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            query_id, _, doc_id, _, score, _ = line.strip().split()
            runs[query_id].append((doc_id, float(score)))
    return runs


def rrf_fuse_single_query(bm25_list, hnsw_list):
    """
    Reciprocal Rank Fusion (RRF) for a single query, given 2 separately ranked lists

    Each list is [(doc_id, score)] sorted by rank 

    Returns:
    fused ranking of [(doc_id, rrf_score)]
    """    
    scores = {}

    # RRF scores from BM25
    for rank, (doc_id, _) in enumerate(bm25_list, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)

    # RRF scores from HNSW
    for rank, (doc_id, _) in enumerate(hnsw_list, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)

    # Sort documents by aggregated RRF score (descending) and slice top K
    fused_ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:K]
    return fused_ranking


def rrf_fusion(bm25_file, hnsw_file, trec_file):
    """
    RRF for all queries in BM25/HNSW trec runs
    """
    bm25_runs = read_trec_run(bm25_file)
    hnsw_runs = read_trec_run(hnsw_file)

    fused_rankings = {}
    for query_id in bm25_runs.keys():
        bm25_list = bm25_runs.get(query_id, [])
        hnsw_list = hnsw_runs.get(query_id, [])
        fused_rankings[query_id] = rrf_fuse_single_query(bm25_list, hnsw_list)
    
    write_trec_run(trec_file, fused_rankings)


def write_trec_run(filename, fused_rankings):
    with open(filename, 'w') as f:
        for query_id, ranking in fused_rankings.items():
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} RRF\n")


def main():
    bm25_files = ["bm25.dev.top100.trec", "bm25.eval.one.top100.trec", "bm25.eval.two.top100.trec"]
    hnsw_files = ["hnsw.dev.trec", "hnsw.eval.one.trec", "hnsw.eval.two.trec"]
    trec_files = ["fusion.dev.trec", "fusion.eval.one.trec", "fusion.eval.two.trec"]

    for i in range(len(bm25_files)):
        rrf_fusion(bm25_files[i], hnsw_files[i], trec_files[i])


if __name__ == "__main__":
    main()