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

# 3. CONFIGURE GEMINI 2.5 FLASH
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
def read_root():
    """This prevents the 'Not Found' error when visiting the base URL."""
    return """
    <html>
        <head><title>Fact Checker API</title></head>
        <body>
            <h1>Fact Checker API is Active</h1>
            <p>Send a POST request to <code>/extract-and-verify</code> or visit 
            <a href="/docs">/docs</a> for the interactive UI.</p>
        </body>
    </html>
    """

def extract_claims(text: str) -> list:
    try:
        # Gemini 2.5 Flash is stable and has high limits
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
            
            # Step 2: Fetch Wikipedia
            search_results = wikipedia.search(search_query, results=1)
            if not search_results:
                results.append({"claim": claim, "found": "No matching Wikipedia article."})
                continue

            page = wikipedia.page(search_results[0], auto_suggest=False)
            
            # Step 3: Direct Summary (No judgments like Confirmed/Refuted)
            verify_prompt = f"""
            Claim: {claim}
            Wiki Evidence: {page.summary[:1500]}
            Based on the evidence, what does Wikipedia say? 
            One clear sentence. Do not use labels like 'Confirmed' or 'Refuted'.
            """
            
            # Small delay for safety (2.5 Flash tier handles this easily)
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
    # uvicorn.run must be at the BOTTOM so all routes are registered first
    uvicorn.run(app, host="0.0.0.0", port=8000)
