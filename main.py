from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
import logging
import google.generativeai as genai
import os
import re
import time

# 1. SETUP LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. INITIALIZE APP
app = FastAPI(title="Wikipedia Fact Checker API")

# 3. CONFIGURE GEMINI 2.5 FLASH (Stable Tier)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found. API calls will fail (or be mocked).")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

# --- ROUTES ---

def _load_index_html() -> str:
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>Fact Checker API is Running</h1>
        <p>Error: <b>index.html</b> not found in the script directory.</p>
        <p>You can still test the API via <a href="/docs">/docs</a></p>
        """


@app.get("/", response_class=HTMLResponse)
def serve_interface_root():
    """Serves the UI at the root URL to prevent 404s."""
    return _load_index_html()


@app.get("/app", response_class=HTMLResponse)
def serve_interface_app():
    """Serves the UI at /app for Render or other app-style routes."""
    return _load_index_html()

def extract_claims(text: str) -> list:
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = (
            "Extract the top 5 factual claims from this text. "
            "Make each claim standalone by resolving pronouns. "
            "Return ONLY a numbered list:\n\n"
            f"{text}"
        )
        response = model.generate_content(prompt)
        claims = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                claim = line.lstrip('0123456789.-* ').strip()
                if claim:
                    claims.append(claim)
        return claims
    except Exception as e:
        logger.error(f"Extraction Error: {e}")
        return []

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "he", "her", "him", "his", "i", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "they", "this", "to", "was", "were",
    "will", "with", "you", "your"
}

_IRRELEVANT_SECTION_TITLES = {
    "see also",
    "notes",
    "references",
    "bibliography",
    "further reading",
    "external links",
    "sources",
    "works",
    "publications"
}


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\bU\.S\.A?\.?\b", "USA", text)


def _trim_irrelevant_sections(content: str) -> str:
    if not content:
        return ""
    lines = content.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        heading = re.match(r"^=+\s*(.*?)\s*=+$", stripped)
        if heading:
            title = heading.group(1).strip().lower()
            if title in _IRRELEVANT_SECTION_TITLES:
                break
        kept.append(line)
    return "\n".join(kept)


def _clean_wikipedia_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[[^\]]+\]", "", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("*", "|", "!", "{{", "}}", "File:", "Image:")):
            continue
        if stripped.startswith("="):
            continue
        lines.append(stripped)
    cleaned = " ".join(lines)
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokenize(text: str) -> set:
    normalized = _normalize_text(text.lower())
    tokens = re.findall(r"[A-Za-z0-9']+", normalized)
    return {t for t in tokens if t and t not in _STOPWORDS}

def _split_sentences(text: str) -> list:
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _build_sentence_candidates(page) -> tuple:
    summary = _clean_wikipedia_text(getattr(page, "summary", "") or "")
    content = _clean_wikipedia_text(_trim_irrelevant_sections(getattr(page, "content", "") or ""))
    summary_sentences = _split_sentences(summary)
    content_sentences = _split_sentences(content)

    sentences = []
    seen = set()
    summary_count = 0

    for sentence in summary_sentences:
        norm = re.sub(r"\s+", " ", sentence).strip().lower()
        if not norm or norm in seen:
            continue
        if len(_tokenize(sentence)) < 2:
            continue
        seen.add(norm)
        sentences.append(sentence)
        summary_count += 1

    for sentence in content_sentences:
        norm = re.sub(r"\s+", " ", sentence).strip().lower()
        if not norm or norm in seen:
            continue
        if len(_tokenize(sentence)) < 2:
            continue
        seen.add(norm)
        sentences.append(sentence)

    return sentences, summary_count


def _extract_phrases(text: str) -> set:
    normalized = _normalize_text(text.lower())
    tokens = [t for t in re.findall(r"[A-Za-z0-9']+", normalized) if t and t not in _STOPWORDS]
    phrases = set()
    for size in (2, 3):
        for idx in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[idx:idx + size])
            if len(phrase) >= 5:
                phrases.add(phrase)
    return phrases


def _object_entity_tokens(claim: str) -> set:
    entities = _extract_entities(claim)
    if len(entities) <= 1:
        return set()
    return _tokenize(" ".join(entities[1:]))


def _required_keyword_groups(claim: str) -> list:
    normalized = _normalize_text(claim.lower())
    groups = []
    object_tokens = _object_entity_tokens(claim)

    if re.search(r"\b(from|born|birth|native|nationality)\b", normalized):
        groups.append({"born", "birth", "birthplace", "native"})
        if object_tokens:
            groups.append(object_tokens)

    if re.search(r"\b(move|moved|emigrate|emigrated|immigrate|immigrated|relocate|relocated|settled|arrived)\b", normalized):
        groups.append({"move", "moved", "emigrate", "emigrated", "immigrate", "immigrated", "relocate", "relocated", "settled", "arrived"})
        if object_tokens:
            groups.append(object_tokens)

    if re.search(r"\blived\b.*\blife\b|\brest of (his|her|their) life\b|\bspent\b.*\b(later|final)\b", normalized):
        groups.append({"lived", "life", "rest", "final", "later", "years", "died", "death", "resident", "resided"})
        if object_tokens:
            groups.append(object_tokens)

    if re.search(r"\b(never )?met\b|\bmeet\b|\bmeeting\b", normalized):
        groups.append({"met", "meet", "meeting", "encountered", "corresponded", "correspondence", "congratulations"})
        if object_tokens:
            groups.append(object_tokens)

    if re.search(r"\b(discover|invent|develop|create)\b", normalized):
        groups.append({"discover", "discovered", "invent", "invented", "developed", "created"})
        if object_tokens:
            groups.append(object_tokens)

    return [g for g in groups if g]


def _noise_penalty(sentence: str) -> int:
    if not sentence:
        return 0
    penalty = 0
    lower = sentence.lower()
    if len(sentence) > 280:
        penalty += 2
    if len(sentence) > 400:
        penalty += 2
    if sentence.count(";") >= 2:
        penalty += 2
    if sentence.count(",") >= 6:
        penalty += 1
    if re.search(r"\b(isbn|oclc|doi|retrieved|archived|citation|href|http|https)\b", lower):
        penalty += 3
    if re.search(r"\b(see also|references|bibliography|external links)\b", lower):
        penalty += 3
    return penalty


def _position_bonus(index: int, summary_count: int) -> int:
    if index < summary_count:
        return 3
    if index < summary_count + 3:
        return 2
    if index < summary_count + 10:
        return 1
    return 0


def _score_sentence(
    sentence: str,
    claim_tokens: set,
    entity_tokens: set,
    keyword_tokens: set,
    require_keyword: bool,
    required_groups: list,
    phrases: set
) -> int:
    if not sentence:
        return 0
    sent_tokens = _tokenize(sentence)
    if not sent_tokens:
        return 0
    if required_groups:
        for group in required_groups:
            if group and not (sent_tokens & group):
                return 0
    keyword_overlap = len(sent_tokens & keyword_tokens)
    if require_keyword and keyword_tokens and keyword_overlap == 0:
        return 0
    overlap = len(sent_tokens & claim_tokens)
    entity_overlap = len(sent_tokens & entity_tokens)
    phrase_bonus = 0
    normalized_sentence = _normalize_text(sentence.lower())
    for phrase in phrases:
        if phrase in normalized_sentence:
            phrase_bonus += 2
    length_penalty = max(0, (len(sent_tokens) - 30) // 8)
    return (
        overlap
        + (2 * entity_overlap)
        + (2 * keyword_overlap)
        + phrase_bonus
        - length_penalty
        - _noise_penalty(sentence)
    )


def _extract_relevant_sentences(page, claim: str, max_sentences: int = 4) -> tuple:
    claim_tokens = _tokenize(claim)
    entity_tokens = _tokenize(" ".join(_extract_entities(claim)))
    keyword_tokens = _expand_claim_keywords(claim)
    required_groups = _required_keyword_groups(claim)
    phrases = _extract_phrases(claim)
    if not claim_tokens:
        return [], 0

    sentences, summary_count = _build_sentence_candidates(page)
    scored = _score_sentences(
        sentences,
        claim_tokens,
        entity_tokens,
        keyword_tokens,
        True,
        required_groups,
        phrases,
        summary_count
    )

    if not scored:
        scored = _score_sentences(
            sentences,
            claim_tokens,
            entity_tokens,
            keyword_tokens,
            False,
            required_groups,
            phrases,
            summary_count
        )

    if not scored:
        fallback = _fallback_sentences(sentences, entity_tokens, max_sentences)
        fallback_score = _fallback_score(fallback, entity_tokens)
        return fallback, fallback_score

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:max_sentences]
    selected.sort(key=lambda item: item[1])
    evidence = [item[2] for item in selected]
    total_score = sum(item[0] for item in selected)
    return evidence, total_score


def _score_sentences(
    sentences: list,
    claim_tokens: set,
    entity_tokens: set,
    keyword_tokens: set,
    require_keyword: bool,
    required_groups: list,
    phrases: set,
    summary_count: int
) -> list:
    scored = []
    for idx, sentence in enumerate(sentences):
        score = _score_sentence(
            sentence,
            claim_tokens,
            entity_tokens,
            keyword_tokens,
            require_keyword,
            required_groups,
            phrases
        )
        if score > 0:
            score += _position_bonus(idx, summary_count)
            scored.append((score, idx, sentence))
    return scored


def _fallback_sentences(sentences: list, entity_tokens: set, max_sentences: int) -> list:
    if not sentences:
        return []

    if entity_tokens:
        scored = []
        for idx, sentence in enumerate(sentences):
            sent_tokens = _tokenize(sentence)
            score = len(sent_tokens & entity_tokens)
            scored.append((score, idx, sentence))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[:max_sentences]
        selected.sort(key=lambda item: item[1])
        return [item[2] for item in selected]

    return sentences[:max_sentences]


def _fallback_score(sentences: list, entity_tokens: set) -> int:
    if not sentences:
        return 0
    if not entity_tokens:
        return 1
    total = 0
    for sentence in sentences:
        total += len(_tokenize(sentence) & entity_tokens)
    return total


def _expand_claim_keywords(claim: str) -> set:
    normalized = _normalize_text(claim.lower())
    keywords = set(_tokenize(normalized))

    if re.search(r"\b(from|born|birth|native|nationality)\b", normalized):
        keywords.update({"born", "birth", "birthplace", "native", "village", "city", "country"})

    if re.search(r"\b(move|moved|emigrate|emigrated|immigrate|immigrated|relocate|relocated|settled|arrived)\b", normalized):
        keywords.update({"moved", "emigrated", "immigrated", "immigration", "emigration", "relocated", "settled", "arrived"})
        keywords.update({"united", "states", "usa", "america", "american", "us"})

    if re.search(r"\blived\b.*\blife\b|\brest of (his|her|their) life\b|\bspent\b.*\b(later|final)\b", normalized):
        keywords.update({"lived", "life", "rest", "final", "later", "years", "died", "death", "resident", "resided"})
        keywords.update({"united", "states", "usa", "america", "american", "us", "new", "york", "ny"})

    if re.search(r"\bnever\b.*\bmet\b|\bdid not\b.*\bmeet\b|\bdidn't\b.*\bmeet\b", normalized):
        keywords.update({"met", "meet", "meeting", "encountered", "corresponded", "correspondence"})
        keywords.update({"wrote", "letter", "letters", "congratulations", "birthday", "greeted"})

    if re.search(r"\b(ac|alternating current)\b", normalized):
        keywords.update({"alternating", "current", "ac", "induction", "motor", "power", "system"})

    if re.search(r"\b(discover|invent|develop|create)\b", normalized):
        keywords.update({"discover", "discovered", "invent", "invented", "developed", "created", "patent", "patents"})

    if "croatia" in normalized:
        keywords.update({"croatia", "smiljan", "austrian", "empire", "serb", "serbian"})

    return keywords

def _extract_entities(text: str) -> list:
    pattern = r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b"
    matches = re.findall(pattern, text)
    ignore = {"The", "A", "An", "And", "Or", "But", "Wikipedia"}
    entities = [m.strip() for m in matches if m.strip() and m.strip() not in ignore]
    return entities


def _build_search_query(claim: str) -> str:
    entities = _extract_entities(claim)
    if len(entities) >= 2:
        return f"{entities[0]} {entities[1]}"
    if len(entities) == 1:
        return entities[0]
    return claim


def _pick_best_title(titles: list, query: str) -> str:
    if not titles:
        return ""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return titles[0]
    best_title = titles[0]
    best_score = None
    query_lower = query.lower().strip()
    for title in titles:
        title_tokens = _tokenize(title)
        overlap = len(query_tokens & title_tokens)
        exact = 1 if title.lower() == query_lower else 0
        prefix = 1 if title.lower().startswith(query_lower) else 0
        extra_tokens = max(len(title_tokens - query_tokens), 0)
        score = (exact, prefix, overlap, -extra_tokens, -len(title))
        if best_score is None or score > best_score:
            best_score = score
            best_title = title
    return best_title


def _best_title_for_query(query: str) -> str:
    search_results = wikipedia.search(query, results=5)
    if not search_results:
        return ""
    return _pick_best_title(search_results, query)


def _select_candidate_titles(claim: str) -> list:
    entities = _extract_entities(claim)
    candidates = []

    if entities:
        primary = entities[0]
        primary_title = _best_title_for_query(primary)
        if primary_title:
            candidates.append(primary_title)

        for ent in entities[1:]:
            title = _best_title_for_query(ent)
            if title:
                candidates.append(title)

    fallback_title = _best_title_for_query(claim)
    if fallback_title:
        candidates.append(fallback_title)

    seen = set()
    ordered = []
    for title in candidates:
        if title not in seen:
            seen.add(title)
            ordered.append(title)
    return ordered


def _fetch_page(title: str, query: str):
    try:
        return wikipedia.page(title, auto_suggest=False)
    except DisambiguationError as e:
        option_title = _pick_best_title(e.options[:10], query)
        return wikipedia.page(option_title, auto_suggest=False)
    except PageError:
        return wikipedia.page(title, auto_suggest=True)


@app.post("/extract-and-verify")
async def extract_and_verify(request: Request):
    """
    Accepts either:
    - JSON: {"text": "..."}
    - form-data with a field named 'text' (fallback)
    This avoids 404/422 when the page posts form-encoded data.
    """
    # Read request body as JSON if possible, otherwise try form
    text = None
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            text = payload.get("text")
    except Exception:
        # not JSON; try form
        try:
            form = await request.form()
            text = form.get("text")
        except Exception:
            text = None

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request body (JSON or form field).")

    logger.info("Received text for verification: %s", text[:120])

    claims = extract_claims(text)
    if not claims:
        return {"claims_found": 0, "results": []}

    model = genai.GenerativeModel('gemini-2.5-flash')
    results = []

    for claim in claims:
        try:
            # Step 1: Select candidate pages (primary entity first)
            candidate_titles = _select_candidate_titles(claim)
            if not candidate_titles:
                results.append({"claim": claim, "found": "No matching Wikipedia article."})
                continue

            # Step 2: Extract evidence from the primary page first, then fall back
            selected_page = None
            selected_evidence = []
            selected_score = 0
            min_score = 2

            for title in candidate_titles:
                page = _fetch_page(title, claim)
                evidence_sentences, score = _extract_relevant_sentences(page, claim, max_sentences=3)
                if evidence_sentences and score >= min_score:
                    selected_page = page
                    selected_evidence = evidence_sentences
                    selected_score = score
                    break
                if score > selected_score:
                    selected_page = page
                    selected_evidence = evidence_sentences
                    selected_score = score

            if not selected_page:
                results.append({"claim": claim, "found": "No matching Wikipedia article."})
                continue

            evidence_text = " ".join(selected_evidence) if selected_evidence else ""

            if not evidence_text:
                verification_res = "No evidence sentences available to summarize."
            else:
                # Step 4: Direct Summary based on evidence (no claim judgement)
                verify_prompt = f"""
                Evidence sentences: {evidence_text}
                Write one concise sentence summarizing only what the evidence says.
                Do not evaluate the claim and do not mention whether it is supported.
                Do not refer to the claim explicitly.
                """

                # Small delay to respect rate limits
                time.sleep(0.5)

                verification_res = model.generate_content(verify_prompt).text.strip()

            results.append({
                "claim": claim,
                "found": verification_res,
                "evidence": selected_evidence,
                "source": selected_page.url
            })

        except Exception as e:
            logger.exception("Error processing claim: %s", claim)
            results.append({"claim": claim, "found": "Data retrieval error."})

    return {"claims_found": len(claims), "results": results}

# --- STARTUP ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
