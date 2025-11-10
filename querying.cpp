#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cctype>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <unordered_set>

using namespace std;

const double N = 1000000;
const double k1 = 1.2;
const double b = 0.75;
const int k = 1000;

struct BlockMetadata
{
    uint32_t lastDocId;
    uint32_t docSize;  // compressed doc size
    uint32_t freqSize; // compressed freq size
};

struct LexiconEntry
{
    uint32_t startBlock; // which block term starts in
    uint32_t startIndex; // which index within block term start (0-127)
    uint32_t listLength; // total postings for the term
};

struct ScoreDoc
{
    double score;
    uint32_t docId;
};

struct MinHeapComp
{ // functor for pq comparator
    bool operator()(const ScoreDoc &a, const ScoreDoc &b) const
    {
        if (a.score == b.score)
        {
            return a.docId > b.docId;
        }
        else
        {
            return a.score > b.score;
        }
    }
};

class ListPointer
{
public:
    ListPointer(const string &term, const LexiconEntry &lexicon) : term(term), listLength(lexicon.listLength), blockNum(lexicon.startBlock), startBlock(lexicon.startBlock), startIndex(lexicon.startIndex)
    {
        uint32_t postingsLeft = (lexicon.listLength > (128 - lexicon.startIndex)) ? (lexicon.listLength - (128 - lexicon.startIndex)) : 0;
        finalBlock = lexicon.startBlock + (postingsLeft + 127) / 128;
    }

    // load 1 block of docIDs and freqs into ListPointer buffers
    void loadBlock(ifstream &ifs, const vector<BlockMetadata> &metadata, const vector<uint64_t> &blockOffsets)
    {
        if (blockNum >= metadata.size())
        {
            return;
        }

        // seek and read compressed bytes from compressed inverted index at offset into buffer
        ifs.clear();
        ifs.seekg(blockOffsets[blockNum], ios::beg);

        // compressed doc bytes
        uint32_t docSize = metadata[blockNum].docSize;
        docBuffer.resize(docSize);
        ifs.read(reinterpret_cast<char *>(docBuffer.data()), docSize);

        // compressed freq bytes
        uint32_t freqSize = metadata[blockNum].freqSize;
        freqBuffer.resize(freqSize);
        ifs.read(reinterpret_cast<char *>(freqBuffer.data()), freqSize);

        docBufPos = 0;
        freqBufPos = 0;
        prevDocId = 0; // reset delta base when new block starts

        // skip docIDs before startIndex
        if (blockNum == startBlock && startIndex > 0)
        {
            for (uint32_t i = 0; i < startIndex; ++i)
            {
                uint32_t gap = varbyteDecode(docBuffer, docBufPos);
                prevDocId += gap;
                varbyteDecode(freqBuffer, freqBufPos);
            }
        }

        currentPos = 0;
    }

    uint32_t nextGEQ(uint32_t targetDoc, ifstream &ifs, const vector<BlockMetadata> &metadata, const vector<uint64_t> &blockOffsets)
    {
        if (currentPos >= listLength)
        { // exhausted this term's postings
            return UINT32_MAX;
        }

        // linear decoding one by one
        while (true)
        {
            if (docBufPos >= docBuffer.size()) // need new block
            {
                if (++blockNum > finalBlock || blockNum >= metadata.size())
                    return UINT32_MAX;
                loadBlock(ifs, metadata, blockOffsets);
            }

            uint32_t gap = varbyteDecode(docBuffer, docBufPos);
            uint32_t doc = prevDocId + gap;
            prevDocId = doc;

            uint32_t freq = varbyteDecode(freqBuffer, freqBufPos);

            ++currentPos;
            currentDoc = doc;
            currentFreq = freq;

            if (doc >= targetDoc)
                return doc;
        }
    }

    double getScore(double docLength, double averageDocLength) const
    {
        // BM25
        double logNum = N - listLength + 0.5;
        double logDenom = listLength + 0.5;
        double operand_one = log(logNum / logDenom);

        double normalized_doc = docLength / averageDocLength;
        double big_k = k1 * ((1 - b) + (b * normalized_doc));
        double num = (k1 + 1) * currentFreq;
        double denom = big_k + currentFreq;
        double operand_two = num / denom;

        double score = operand_one * operand_two;
        return score;
    }

    void close()
    {
        docBuffer.clear();
        freqBuffer.clear();
    }

    // needed to get maxscore approx
    uint32_t getListLength() const
    {
        return listLength;
    }

    void setCurrentFrequency(uint32_t val)
    {
        currentFreq = val;
    }

private:
    uint32_t varbyteDecode(const vector<unsigned char> &buf, size_t &pos)
    {
        uint32_t num = 0;
        uint32_t shift = 0;
        uint8_t curr;

        // varbyte is little endian, decode one num at a time
        do
        {
            curr = buf[pos++];
            num += (curr & 127) << shift;
            shift += 7;
        } while (curr >= 128);

        return num;
    }

    string term;
    uint32_t listLength;     // total postings for term
    uint32_t currentPos = 0; // curr index in postings list
    uint32_t currentDoc;     // most recent decoded docID, updated on nextGEQ
    uint32_t currentFreq;    // freq of term in currentDoc
    uint32_t blockNum;       // index of current COMPRESSED block in file (based on startBlock)
    uint32_t finalBlock;     // prevents galloping from bleeding into next term's postings
    uint32_t startBlock;     // first block where term inverted list starts
    uint32_t startIndex;     // first index offset within start block
    uint32_t prevDocId;      // for delta decoding varbyte
    // buffers for curr block (compressed)
    vector<unsigned char> docBuffer;  // to store compressed bytes from disk, read metadata[blockNum].docSize bytes
    vector<unsigned char> freqBuffer; // to store compressed bytes from disk, read metadata[blockNum].freqSize bytes
    // will decompress byte-by-byte when we need posting

    // position inside compressed buffers (byte offset)
    size_t docBufPos = 0;  // byte position inside current docBuffer, reset to 0 when load new block, indicates how far in buffer decoded, updated when decoding
    size_t freqBufPos = 0; // byte position inside current freqBuffer, reset to 0 when load new block, indicates how far in buffer decoded, updated when decoding
};

vector<uint64_t> computeBlockOffsets(const vector<BlockMetadata> &metadata);
vector<ScoreDoc> disjunctiveDAAT(const vector<string> &queryTerms,
                                 const unordered_map<string, size_t> &termToIndex,
                                 ifstream &ifs,
                                 const vector<LexiconEntry> &lexicon,
                                 const vector<BlockMetadata> &metadata,
                                 const vector<uint64_t> &blockOffsets,
                                 unordered_map<int, int> &pageTable,
                                 double averageDocLength);
unordered_map<int, int> loadPageTable(ifstream &ifs);
double getAverageDocLength(const unordered_map<int, int> &pageTable);
vector<LexiconEntry> loadLexicon(ifstream &ifs, unordered_map<string, size_t> &termToIndex);
vector<BlockMetadata> loadMetadata(ifstream &ifs);
unordered_map<uint32_t, string> loadActualQueries(ifstream &ifs);
void writeTrecResults(ofstream &ofs, uint32_t queryId, const vector<ScoreDoc> &rankedDocs, size_t k);
void cleanQuery(string &query);
vector<ScoreDoc> processQuery(const string &query,
                              uint32_t queryId,
                              unordered_map<string, size_t> &termToIndex,
                              ifstream &indexIfs,
                              const vector<LexiconEntry> &lexicon,
                              const vector<BlockMetadata> &metadata,
                              const vector<uint64_t> &blockOffsets,
                              unordered_map<int, int> &pageTable,
                              double averageDocLength);

int main()
{
    // compressed index, lexicon, metadata, page table
    string indexFilename = "compressed_inverted_index.bin";
    string lexiconFilename = "lexicon.bin";
    string metadataFilename = "metadata.bin";
    string pageTableFilename = "page_table.txt";
    ifstream indexIfs(indexFilename, ios::binary);
    ifstream lexiconIfs(lexiconFilename, ios::binary);
    ifstream metadataIfs(metadataFilename, ios::binary);
    ifstream pageTableIfs(pageTableFilename);

    if (!indexIfs || !lexiconIfs || !metadataIfs || !pageTableIfs)
    {
        cerr << "Failed to open files!" << endl;
        return 1;
    }

    // put page table in memory
    unordered_map<int, int> pageTable = loadPageTable(pageTableIfs);
    double averageDocLength = getAverageDocLength(pageTable);

    // put lexicon in memory and have mapping from term to index
    unordered_map<string, size_t> termToIndex;
    vector<LexiconEntry> lexicon = loadLexicon(lexiconIfs, termToIndex);

    // process metadata in memory
    vector<BlockMetadata> metadata = loadMetadata(metadataIfs);
    vector<uint64_t> blockOffsets = computeBlockOffsets(metadata);

    // get query
    string queryInput;
    // get input from the qrels.dev.tsv and qrels.eval.tsv
    string devQueries = "qrels.dev.tsv";
    string evalOneQueries = "qrels.eval.one.tsv";
    string evalTwoQueries = "qrels.eval.two.tsv";
    string devActualQueries = "queries.dev.tsv";
    string evalActualQueries = "queries.eval.tsv";

    ifstream devIfs(devQueries);
    ifstream evalOneIfs(evalOneQueries);
    ifstream evalTwoIfs(evalTwoQueries);
    ifstream devActualIfs(devActualQueries);
    ifstream evalActualIfs(evalActualQueries);

    if (!devIfs || !evalOneIfs || !evalTwoIfs || !devActualIfs || !evalActualIfs)
    {
        cerr << "Failed to open files!" << endl;
        return 1;
    }

    unordered_map<uint32_t, string> devQueryMap = loadActualQueries(devActualIfs);
    unordered_map<uint32_t, string> evalQueryMap = loadActualQueries(evalActualIfs);

    string query;
    string line;
    int ignore;
    uint32_t queryId;
    uint32_t passageId;
    uint8_t relevance;                               // 0-3
    vector<pair<uint32_t, vector<ScoreDoc>>> buffer; // stpre 100 queries at a time
    unordered_set<uint32_t> uniqueQueries;
    uint32_t counter = 0;

    string devTrecTop100Filename = "bm25.dev.top100.trec";
    string devTrecTop1000Filename = "bm25.dev.top1000.trec";

    string evalOneTrecTop100Filename = "bm25.eval.one.top100.trec";
    string evalOneTrecTop1000Filename = "bm25.eval.one.top1000.trec";

    string evalTwoTrecTop100Filename = "bm25.eval.two.top100.trec";
    string evalTwoTrecTop1000Filename = "bm25.eval.two.top1000.trec";

    // qrels.dev.tsv
    cout << "Processing qrels.dev.tsv" << endl;
    ofstream ofsTop100(devTrecTop100Filename);
    ofstream ofsTop1000(devTrecTop1000Filename);
    while (getline(devIfs, line))
    {
        stringstream ss(line);
        ss >> queryId >> passageId >> relevance;
        uniqueQueries.insert(queryId);
    }

    for (uint32_t queryId : uniqueQueries)
    {
        query = devQueryMap[queryId];
        vector<ScoreDoc> results = processQuery(query, queryId, termToIndex, indexIfs, lexicon, metadata, blockOffsets, pageTable, averageDocLength);

        buffer.push_back({queryId, results});
        ++counter;

        if (counter % 100 == 0)
        {
            for (const auto &entry : buffer)
            {
                writeTrecResults(ofsTop100, entry.first, entry.second, 100);
                writeTrecResults(ofsTop1000, entry.first, entry.second, 1000);
            }
            buffer.clear();
            cout << "Flushed 100 queries to disk." << endl;
        }
    }

    // remaining ones (if < 100 left)
    for (const auto &entry : buffer)
    {
        writeTrecResults(ofsTop100, entry.first, entry.second, 100);
        writeTrecResults(ofsTop1000, entry.first, entry.second, 1000);
    }

    buffer.clear();
    uniqueQueries.clear();
    ofsTop100.close();
    ofsTop1000.close();
    cout << "Flushed final queries to disk." << endl;

    // qrels.eval.one.tsv
    ofsTop100.open(evalOneTrecTop100Filename);
    ofsTop1000.open(evalOneTrecTop1000Filename);
    cout << "Processing " << evalOneQueries << " " << endl;
    while (getline(evalOneIfs, line))
    {
        stringstream ss(line);
        ss >> queryId >> ignore >> passageId >> relevance;
        uniqueQueries.insert(queryId);
    }

    for (uint32_t queryId : uniqueQueries)
    {
        query = evalQueryMap[queryId];
        vector<ScoreDoc> results = processQuery(query, queryId, termToIndex, indexIfs, lexicon, metadata, blockOffsets, pageTable, averageDocLength);
        buffer.push_back({queryId, results});
    }

    for (const auto &entry : buffer)
    {
        writeTrecResults(ofsTop100, entry.first, entry.second, 100);
        writeTrecResults(ofsTop1000, entry.first, entry.second, 1000);
    }

    buffer.clear();
    uniqueQueries.clear();
    ofsTop100.close();
    ofsTop1000.close();
    cout << "Flushed final queries to disk." << endl;

    // qrels.eval.two.tsv
    ofsTop100.open(evalTwoTrecTop100Filename);
    ofsTop1000.open(evalTwoTrecTop1000Filename);
    cout << "Processing " << evalTwoQueries << " " << endl;
    while (getline(evalTwoIfs, line))
    {
        stringstream ss(line);
        ss >> queryId >> ignore >> passageId >> relevance;
        uniqueQueries.insert(queryId);
    }

    for (uint32_t queryId : uniqueQueries)
    {
        query = evalQueryMap[queryId];
        vector<ScoreDoc> results = processQuery(query, queryId, termToIndex, indexIfs, lexicon, metadata, blockOffsets, pageTable, averageDocLength);
        buffer.push_back({queryId, results});
    }

    for (const auto &entry : buffer)
    {
        writeTrecResults(ofsTop100, entry.first, entry.second, 100);
        writeTrecResults(ofsTop1000, entry.first, entry.second, 1000);
    }

    buffer.clear();
    uniqueQueries.clear();
    cout << "Flushed all queries to disk." << endl;

    // close all filestreams
    indexIfs.close();
    lexiconIfs.close();
    metadataIfs.close();
    pageTableIfs.close();
    devIfs.close();
    evalOneIfs.close();
    evalTwoIfs.close();
    devActualIfs.close();
    evalActualIfs.close();
}

// compute block offsets once from metadata for each block instead of doing it each time we get a term
vector<uint64_t> computeBlockOffsets(const vector<BlockMetadata> &metadata)
{
    // basically prefix sums
    vector<uint64_t> offsets(metadata.size());
    uint64_t off = 0;
    for (size_t i = 0; i < metadata.size(); ++i)
    {
        offsets[i] = off;
        off += (uint64_t)metadata[i].docSize + (uint64_t)metadata[i].freqSize;
    }
    return offsets;
}

vector<ScoreDoc> disjunctiveDAAT(const vector<string> &queryTerms,
                                 const unordered_map<string, size_t> &termToIndex,
                                 ifstream &ifs,
                                 const vector<LexiconEntry> &lexicon,
                                 const vector<BlockMetadata> &metadata,
                                 const vector<uint64_t> &blockOffsets,
                                 unordered_map<int, int> &pageTable,
                                 double averageDocLength)
{
    size_t numTerms = queryTerms.size();
    // iterate over union of postings, compute
    vector<ListPointer *> lp(numTerms);

    // open all lists
    for (size_t i = 0; i < numTerms; ++i)
    {
        ListPointer *p = new ListPointer(queryTerms[i], lexicon[termToIndex.at(queryTerms[i])]);
        p->loadBlock(ifs, metadata, blockOffsets);
        lp[i] = p;
    }

    // sort posting lists by max possible impact score to identify essential lists
    // don't want to decode each frequency to find max, so just use listLength to set upper bound
    vector<double> maxScores(numTerms);
    for (size_t i = 0; i < numTerms; ++i)
    {
        // assume highest freq is listLength (max possible freq)
        // approx upper bound for each list
        uint32_t listLength = lp[i]->getListLength();
        lp[i]->setCurrentFrequency(listLength);
        // pageTable[8841709] arbitrarily chosen for length normalization since don't know "true" docId yet
        maxScores[i] = lp[i]->getScore(pageTable[8841709], averageDocLength);
    }

    // sort from lowest to highest impact
    // if sum of remaining maxScores (of higher ones) < threshold, can stop early
    vector<size_t> order(numTerms);
    for (size_t i = 0; i < numTerms; ++i)
    {
        order[i] = i;
    }
    sort(order.begin(), order.end(), [&](size_t a, size_t b)
         { return maxScores[a] < maxScores[b]; });

    // keep track of curr docIDs in each list
    vector<uint32_t> currDoc(numTerms);
    for (size_t i = 0; i < numTerms; ++i)
    {
        currDoc[i] = lp[i]->nextGEQ(0, ifs, metadata, blockOffsets);
    }

    // use min heap so we take out minimum out of the top k in constant time
    priority_queue<ScoreDoc, vector<ScoreDoc>, MinHeapComp> topK;

    while (true)
    {
        // find next candidate docID = min of curr docIDs across all term lists
        uint32_t candidate = UINT32_MAX;
        for (size_t i = 0; i < numTerms; ++i)
        {
            if (currDoc[i] != UINT32_MAX)
            {
                candidate = min(candidate, currDoc[i]);
            }
        }
        if (candidate == UINT32_MAX)
        {
            break; // all lists exhausted
        }

        // sum score for candidate
        double score = 0.0;
        double remainingMax = 0.0; // for non-essential early termination, sum of unused maxScores

        for (size_t idx : order)
        {
            // if one of it matches, can add to score, not necessarily all inverted lists need to have it, so we put those in remainingMax
            if (currDoc[idx] == candidate)
            {
                score += lp[idx]->getScore(pageTable[candidate], averageDocLength);

                // advance list to meet >= candidate + 1, so basically next docID
                currDoc[idx] = lp[idx]->nextGEQ(candidate + 1, ifs, metadata, blockOffsets);
            }
            else
            {
                remainingMax += maxScores[idx];
            }
        }

        // early termination: skip non-essential lists if cannot affect topK
        // heap full and if add best possible scores from remaining list, still below threshold, skip it
        if (topK.size() >= k && score + remainingMax <= topK.top().score)
            continue;

        // maintain top-k heap
        if (topK.size() < k)
        {
            topK.push({score, candidate});
        }
        // || (score == topK.top().score && candidate > topK.top().docId) - ignore
        else if (score > topK.top().score)
        {
            topK.pop();
            topK.push({score, candidate});
        }
    }

    for (size_t idx = 0; idx < lp.size(); ++idx)
    {
        lp[idx]->close();
        delete lp[idx];
    }

    vector<ScoreDoc> results;
    while (!topK.empty())
    {
        results.push_back(topK.top());
        topK.pop();
    }
    return results;
}

unordered_map<int, int> loadPageTable(ifstream &ifs)
{
    unordered_map<int, int> table;
    int docId;
    int docLength;
    while (ifs >> docId >> docLength)
    {
        table[docId] = docLength;
    }
    return table;
}

double getAverageDocLength(const unordered_map<int, int> &pageTable)
{
    uint64_t total = 0;
    for (const auto &entry : pageTable)
    {
        total += entry.second;
    }
    return static_cast<double>(total) / pageTable.size();
}

vector<LexiconEntry> loadLexicon(ifstream &ifs, unordered_map<string, size_t> &termToIndex)
{
    vector<LexiconEntry> lexicon;
    uint32_t termSize;
    while (ifs.read(reinterpret_cast<char *>(&termSize), sizeof(termSize)))
    {
        string term(termSize, '\0');
        ifs.read(&term[0], termSize);

        LexiconEntry entry;
        ifs.read(reinterpret_cast<char *>(&entry), sizeof(LexiconEntry));

        termToIndex[term] = lexicon.size();
        lexicon.push_back(entry);
    }
    return lexicon;
}

vector<BlockMetadata> loadMetadata(ifstream &ifs)
{
    vector<BlockMetadata> metadata;
    BlockMetadata block;
    while (ifs.read(reinterpret_cast<char *>(&block), sizeof(BlockMetadata)))
    {
        metadata.push_back(block);
    }
    return metadata;
}

void writeTrecResults(ofstream &ofs, uint32_t queryId, const vector<ScoreDoc> &rankedDocs, size_t k)
{
    uint32_t rank = 1;
    for (const ScoreDoc &entry : rankedDocs)
    {
        if (rank > k)
        {
            break; // only write top k
        }
        ofs << queryId << " Q0 " << entry.docId << " " << rank << " "
            << std::fixed << std::setprecision(6) << entry.score
            << " BM25\n";
        ++rank;
    }
}

void cleanQuery(string &query)
{
    string cleaned;
    for (char c : query)
    {
        unsigned char uc = tolower((unsigned char)c);
        if (uc <= 127 && !ispunct(uc))
        {
            cleaned += uc; // valid non-punctuation ascii
        }
        else
        {
            cleaned += ' '; // replace separating characters with a space to split terms
        }
    }
    query = cleaned;
}

unordered_map<uint32_t, string> loadActualQueries(ifstream &ifs)
{
    unordered_map<uint32_t, string> mapping;
    uint32_t queryId;
    string line;
    string sentence;
    while (getline(ifs, line))
    {
        stringstream ss(line);
        ss >> queryId;
        getline(ss, sentence);
        cleanQuery(sentence);
        mapping[queryId] = sentence;
    }
    return mapping;
}

vector<ScoreDoc> processQuery(const string &query,
                              uint32_t queryId,
                              unordered_map<string, size_t> &termToIndex,
                              ifstream &indexIfs,
                              const vector<LexiconEntry> &lexicon,
                              const vector<BlockMetadata> &metadata,
                              const vector<uint64_t> &blockOffsets,
                              unordered_map<int, int> &pageTable,
                              double averageDocLength)
{
    string term;
    stringstream ss(query);
    vector<string> queryTerms;
    while (ss >> term)
    {
        queryTerms.push_back(term);
    }

    bool atLeastOneFound = false;
    vector<ScoreDoc> results;

    vector<string> foundQueryTerms;
    for (const string &term : queryTerms)
    {
        if (termToIndex.find(term) != termToIndex.end())
        {
            foundQueryTerms.push_back(term);
            // if all terms not found, no results
        }
    }

    if (!foundQueryTerms.empty())
    {
        indexIfs.clear();
        indexIfs.seekg(0, ios::beg);
        results = disjunctiveDAAT(foundQueryTerms, termToIndex, indexIfs, lexicon, metadata, blockOffsets, pageTable, averageDocLength);
    }

    reverse(results.begin(), results.end());
    return results;
}