"""Turn the PDFs in docs/ into passages the question box can search.

    python rag.py            build the index
    python rag.py "repo rate february 2025"    try a search

RAG means retrieval-augmented generation: find the right passages first, then
hand them to the language model along with the question. The model never
searches anything itself -- it only reads what this file selected.

That split is what makes the answers checkable. Every claim traces back to a
passage you can open, and the model's job is reduced to writing English about
text it was given.

Two steps:

  1. Build (slow, run once per batch of documents). Read each PDF page by
     page, cut it into passages, and save them to data/chunks.json with the
     document, date and page number attached. The server never needs the PDFs.

  2. Search (fast, runs on every question). Score every passage against the
     question with BM25 and return the best few.

Why BM25 and not embeddings. BM25 is keyword matching, decades old and
thoroughly boring. It needs no API, no model download, and no cost per query,
and it is very good at exactly the kind of question this project gets -- ones
naming specific things like "repo rate", "February 2025", "fiscal deficit".
Embeddings go in only if the retrieval evals show BM25 missing the right
passage. Making that call from eval numbers rather than fashion is the point.
"""

import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).parent
DOCS = BASE / "docs"
CHUNKS = BASE / "data" / "chunks.json"

# Passage length, in words. Short enough that a passage is about one idea,
# long enough to carry its own context -- a sentence saying "it was raised to
# 5.50%" is useless without the sentence that says what "it" is.
MIN_WORDS = 150
MAX_WORDS = 300

TOP_K = 5


# ---------------------------------------------------------------------------
# Reading the PDFs
# ---------------------------------------------------------------------------

def read_pdf(path):
    """Yield (page_number, text) for each page that has any text on it.

    Pages are kept separate so a citation can name a page. A scanned page with
    no text layer yields nothing -- it is not an error, there is simply nothing
    to index, and the quality count at the end will show it.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        text = re.sub(r"[ \t]+", " ", text)
        if text.strip():
            yield number, text


def document_meta(path):
    """Pull the date and kind out of a filename like 2025-10-01_RBI.pdf."""
    match = re.match(r"(\d{4}-\d{2}-\d{2})_([A-Z]+)", path.stem)
    if not match:
        return None, None
    return match.group(1), match.group(2)


# ===========================================================================
# WRITE THIS ONE  (1 of 2)
# ===========================================================================

def split_into_passages(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    r"""Cut one page of text into passages of roughly min..max words.

    This is the decision that quietly determines whether retrieval works.

    Too short and a passage loses the context that makes it meaningful -- a
    line reading "this was raised by 25 basis points" tells the model nothing
    about what "this" is. Too long and a passage covers several topics at
    once, so matching one of them drags in four irrelevant paragraphs and
    crowds the useful ones out of the top five.

    Splitting on blank lines first matters because paragraphs are already the
    author's own idea of where one thought ends. Cutting mid-paragraph at a
    fixed word count would slice sentences in half.

    The method:

      1. Split `text` on blank lines into paragraphs, dropping empty ones.
         A blank line in extracted PDF text is usually "\n\n", but can have
         stray spaces, so split on r"\n\s*\n".
      2. Walk the paragraphs, collecting them into a buffer and counting
         words as you go.
      3. When the buffer reaches `min_words`, emit it as one passage (join
         the collected paragraphs with "\n") and start a new buffer.
      4. If a single paragraph is longer than `max_words` on its own, split
         that paragraph into `max_words`-sized pieces rather than emitting
         one enormous passage.
      5. At the end, emit whatever is left in the buffer, as long as it has
         at least 20 words. Shorter than that is usually a page header or a
         stray footnote, not a thought.

    Returns
    -------
    list of str
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    passages,buffer,count=[],[],0
    for para in paragraphs:
        words=para.split()
        if len(words)>max_words:
            if buffer:
                passages.append("\n".join(buffer))
                buffer,count=[],0
            for i in range(0,len(words),max_words):
                passages.append(" ".join(words[i:i+max_words]))
            continue
        buffer.append(para)
        count+=len(words)
        if count>=min_words:
                
                passages.append("\n".join(buffer))
                buffer,count=[],0

    if buffer and count>=20:
        passages.append("\n".join(buffer))
    return passages




def build_index():
    """Read every PDF in docs/ and write data/chunks.json."""
    pdfs = sorted(p for p in DOCS.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs in {DOCS} -- fill in docs/sources.txt and run fetch_docs.py")

    chunks = []
    skipped = []
    for path in pdfs:
        date, kind = document_meta(path)
        if not date:
            skipped.append(f"{path.name} (filename is not YYYY-MM-DD_KIND.pdf)")
            continue

        pages = passages = 0
        for page_number, text in read_pdf(path):
            pages += 1
            for passage in split_into_passages(text):
                chunks.append({
                    "id": f"{path.stem}#p{page_number}#{passages}",
                    "doc": path.name,
                    "date": date,
                    "kind": kind,
                    "page": page_number,
                    "text": passage,
                })
                passages += 1
        print(f"  {path.name:34} {pages:>3} pages  {passages:>4} passages")

    if skipped:
        print("\n  skipped:")
        for s in skipped:
            print(f"    {s}")

    CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS.write_text(json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")
    size = CHUNKS.stat().st_size
    print(f"\n  {len(chunks):,} passages from {len(pdfs)} documents "
          f"-> {CHUNKS.name} ({size/1e6:.1f} MB)")
    return chunks


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

def tokenise(text):
    """Words, lowercased, punctuation dropped.

    The question and the passages must be broken up the same way, or they
    cannot match. "Repo rate" in a question has to reach the same two tokens
    as "repo rate." at the end of a sentence in a document.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class Searcher:
    """BM25 over the passages in data/chunks.json."""

    def __init__(self, chunks=None):
        from rank_bm25 import BM25Okapi

        if chunks is None:
            if not CHUNKS.exists():
                raise FileNotFoundError(
                    f"{CHUNKS} is missing -- run `python rag.py` to build it")
            chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenise(c["text"]) for c in chunks])

    # =======================================================================
    # WRITE THIS ONE  (2 of 2)
    # =======================================================================

    def search(self, question, k=TOP_K):
        """The `k` passages that best match `question`, best first.

        This is the whole retrieval step. Whatever it returns is all the
        language model will ever see -- if the right passage is not in this
        list, no amount of prompting will produce a correct answer. Everything
        downstream depends on these few lines.

        BM25 scores each passage on how many of the question's words it
        contains, weighted so that rare words count for more than common ones.
        A passage containing "repo" scores far better than one containing
        "the", which is what makes it work without understanding meaning.
        
        The method:

          1. Tokenise the question with tokenise().
          2. If there are no tokens, return [] -- an empty or
             punctuation-only question has nothing to match on.
          3. scores = self.bm25.get_scores(tokens)  (a NumPy array, one
             score per passage, in the same order as self.chunks)
          4. Take the indices of the k highest scores. The direct way:
             sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
          5. Build a result list. For each index, copy that chunk's dict and
             add its score as a float, so the eval harness and the page can
             both see how strong the match was.
          6. Drop any result whose score is 0 -- that means not one word of
             the question appeared in the passage, so it is filler, and
             sending filler to the model invites it to use filler.

        Returns
        -------
        list of dict : each with doc, date, kind, page, text and score
        """
        tokens = tokenise(question)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for i in best:
            if scores[i] <= 0:
                continue
            hit = dict(self.chunks[i])
            hit["score"] = float(scores[i])
            results.append(hit)
        return results

    def __len__(self):
        return len(self.chunks)


# A single shared searcher, built on first use. Loading the index and scoring
# the corpus takes a moment, and the web server should pay that once at
# startup rather than on every question.
_SEARCHER = None


def searcher():
    global _SEARCHER
    if _SEARCHER is None:
        _SEARCHER = Searcher()
    return _SEARCHER


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        s = searcher()
        print(f"{len(s):,} passages indexed\n")
        for hit in s.search(query):
            head = f"{hit['doc']} p{hit['page']}  score {hit['score']:.2f}"
            print(f"  {head}\n  {'-' * len(head)}")
            print("  " + hit["text"][:320].replace("\n", " ") + "…\n")
    else:
        build_index()
