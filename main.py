from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import wikipedia
import logging
import google.generativeai as genai
import os
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
    logger.error("GEMINI_API_KEY not found. API calls will fail.")

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

@app.get("/", response_class=HTMLResponse)
def serve_interface():
    """Serves the UI directly at the root URL to prevent 404s."""
    try:
        # Tries to load your html file
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback if index.html is missing
        return """
        <h1>Fact Checker API is Running</h1>
        <p>Error: <b>index.html</b> not found in the script directory.</p>
        <p>You can still test the API via <a href="/docs">/docs</a></p>
        """

def extract_claims(text: str) -> list:
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Extract the top 5 factual claims from this text. Return ONLY a numbered list:\n\n{text}"
        response = model.generate_content(prompt)
        claims = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                claim = line.lstrip('0123456789.-* ').strip()
                if claim: claims.append(claim)
        return claims
    except Exception as e:
        logger.error(f"Extraction Error: {e}")
        return []

@app.post("/extract-and-verify")
def extract_and_verify(request: TextRequest):
    claims = extract_claims(request.text)
    if not claims:
        return {"claims_found": 0, "results": []}

    model = genai.GenerativeModel('gemini-2.5-flash')
    results = []

    for claim in claims:
        try:
            # Step 1: Specific Wiki Query
            search_query_res = model.generate_content(f"Wikipedia search query for: {claim}. Return ONLY the query.")
            search_query = search_query_res.text.strip()
            
            # Step 2: Fetch Wikipedia content
            search_results = wikipedia.search(search_query, results=1)
            if not search_results:
                results.append({"claim": claim, "found": "No matching Wikipedia article."})
                continue

            page = wikipedia.page(search_results[0], auto_suggest=False)
            
            # Step 3: Direct Summary (No judgements like Confirmed/Refuted)
            verify_prompt = f"""
            Claim: {claim}
            Wiki Evidence: {page.summary[:1500]}
            Summarize what Wikipedia says about this claim in one sentence. 
            Do not use labels like 'Confirmed' or 'Refuted'.
            """
            
            # 0.5s delay to stay safe under Gemini 2.5 stable rate limits
            time.sleep(0.5)
            
            verification_res = model.generate_content(verify_prompt).text.strip()
            
            results.append({
                "claim": claim,
                "found": verification_res,
                "source": page.url
            })

        except Exception as e:
            logger.error(f"Error on claim '{claim}': {e}")
            results.append({"claim": claim, "found": "Data retrieval error."})

    return {"claims_found": len(claims), "results": results}

# --- STARTUP ---
if __name__ == "__main__":
    import uvicorn
    # uvicorn.run MUST be at the bottom so all routes are registered first
    uvicorn.run(app, host="0.0.0.0", port=8000)
