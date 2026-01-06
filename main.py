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

def _tokenize(text: str) -> set:
    return set(re.findall(r"[A-Za-z0-9']+", text.lower()))

def _split_sentences(text: str) -> list:
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _score_sentence(sentence: str, claim_tokens: set, entity_tokens: set) -> int:
    if not sentence:
        return 0
    sent_tokens = _tokenize(sentence)
    if not sent_tokens:
        return 0
    overlap = len(sent_tokens & claim_tokens)
    entity_overlap = len(sent_tokens & entity_tokens)
    return overlap + (2 * entity_overlap)


def _extract_relevant_sentences(page, claim: str, max_sentences: int = 3) -> list:
    claim_tokens = _tokenize(claim)
    entity_tokens = _tokenize(" ".join(_extract_entities(claim)))
    if not claim_tokens:
        return []

    sentences = _split_sentences(getattr(page, "content", "") or "")
    scored = []
    for idx, sentence in enumerate(sentences):
        score = _score_sentence(sentence, claim_tokens, entity_tokens)
        if score > 0:
            scored.append((score, idx, sentence))

    if not scored:
        sentences = _split_sentences(getattr(page, "summary", "") or "")
        for idx, sentence in enumerate(sentences):
            score = _score_sentence(sentence, claim_tokens, entity_tokens)
            if score > 0:
                scored.append((score, idx, sentence))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:max_sentences]
    selected.sort(key=lambda item: item[1])
    return [item[2] for item in selected]


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
    scored = []
    for title in titles:
        score = len(query_tokens & _tokenize(title))
        scored.append((score, title))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


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
            # Step 1: Build a query from entities or fallback to the claim
            search_query = _build_search_query(claim)

            # Step 2: Fetch Wikipedia content
            search_results = wikipedia.search(search_query, results=5)
            if not search_results:
                search_results = wikipedia.search(claim, results=5)
            if not search_results:
                results.append({"claim": claim, "found": "No matching Wikipedia article."})
                continue

            best_title = _pick_best_title(search_results, search_query)
            try:
                page = wikipedia.page(best_title, auto_suggest=False)
            except DisambiguationError as e:
                option_title = _pick_best_title(e.options[:10], search_query)
                page = wikipedia.page(option_title, auto_suggest=False)
            except PageError:
                page = wikipedia.page(best_title, auto_suggest=True)

            # Step 3: Extract evidence sentences that best match the claim
            evidence_sentences = _extract_relevant_sentences(page, claim, max_sentences=3)
            evidence_text = " ".join(evidence_sentences) if evidence_sentences else ""

            # Step 4: Direct Summary based on evidence (No judgements like Confirmed/Refuted)
            verify_prompt = f"""
            Claim: {claim}
            Evidence sentences: {evidence_text or "No direct evidence found in the article."}
            Summarize what Wikipedia says about this claim in one sentence.
            If the evidence does not address the claim, say that directly.
            Do not use labels like 'Confirmed' or 'Refuted'.
            """

            # Small delay to respect rate limits
            time.sleep(0.5)

            verification_res = model.generate_content(verify_prompt).text.strip()

            results.append({
                "claim": claim,
                "found": verification_res,
                "evidence": evidence_sentences,
                "source": page.url
            })

        except Exception as e:
            logger.exception("Error processing claim: %s", claim)
            results.append({"claim": claim, "found": "Data retrieval error."})

    return {"claims_found": len(claims), "results": results}

# --- STARTUP ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
