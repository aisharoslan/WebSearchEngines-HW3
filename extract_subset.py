import time

def get_subset_ids(filename: str) -> set[int]:
    result = set()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                result.add(line)
        return result

def write_file(filename: str, passage_ids: set[int]) -> None:
    output_filename = "subset_passages.tsv"
    with open(filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('\t', 1)
            if not parts:
                continue
            passage_id = parts[0]
            if passage_id in passage_ids:
                outfile.write(line)
        
def main():
    original_filename = "collection.tsv"
    subset_filename = "msmarco_passages_subset.tsv"
    
    start = time.time()
    passage_ids = get_subset_ids(subset_filename)
    write_file(original_filename, passage_ids)
    end = time.time()

    print(f"Ran in {end - start} seconds")

if __name__ == "__main__":
    main()