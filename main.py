from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse  # Correctly imported
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import wikipedia
import logging
import google.generativeai as genai
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wikipedia Fact Checker API")

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    claim: str

class TextRequest(BaseModel):
    text: str

class VerificationResult(BaseModel):
    status: str 
    claim: str
    evidence: str
    source_url: str
    confidence: str

def extract_claims(text: str) -> list:
    if not GEMINI_API_KEY:
        return []
    try:
        # Using the updated model name for 2025
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        prompt = f"Extract all factual claims from the following text. Return ONLY a numbered list.\n\nText: {text}"
        response = model.generate_content(prompt)
        claims = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                claim = line.lstrip('0123456789.-* ').strip()
                if claim: claims.append(claim)
        return claims
    except Exception as e:
        logger.error(f"Error: {e}")
        return []

@app.get("/")
def read_root():
    return {"message": "Wikipedia Fact Checker API", "interface": "/app"}

# FIXED: Explicitly returns HTMLResponse so the browser renders the page
@app.get("/app", response_class=HTMLResponse)
def serve_app():
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Error: index.html not found in the server directory."

@app.post("/verify-claim", response_model=VerificationResult)
def verify_claim(request: ClaimRequest):
    claim = request.claim.strip()
    try:
        search_results = wikipedia.search(claim, results=1)
        if not search_results:
            return VerificationResult(status="unclear", claim=claim, evidence="No results", source_url="", confidence="low")
        
        page = wikipedia.page(search_results[0], auto_suggest=False)
        # Basic check for claim presence in summary
        status = "confirmed" if claim.lower() in page.summary.lower() else "unclear"
        return VerificationResult(
            status=status,
            claim=claim,
            evidence=page.summary[:500],
            source_url=page.url,
            confidence="medium"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-and-verify")
def extract_and_verify(request: TextRequest):
    claims = extract_claims(request.text)
    results = []
    for c in claims:
        res = verify_claim(ClaimRequest(claim=c))
        # FIXED: Using model_dump() instead of .dict() for Pydantic v2 compatibility
        results.append(res.model_dump()) 
    return {"claims_found": len(claims), "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)