from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        prompt = f"Extract factual claims from this text. Return ONLY a numbered list:\n\n{text}"
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

@app.get("/")
def read_root():
    return {"message": "API is running", "interface": "/app"}

@app.get("/app", response_class=HTMLResponse)
def serve_app():
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Error: index.html not found."

@app.post("/verify-claim", response_model=VerificationResult)
def verify_claim(request: ClaimRequest):
    claim = request.claim.strip()
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        
        # FIX 1: Use AI to create a smart Wikipedia search term instead of just 5 words
        search_prompt = f"Convert this claim into 2-3 specific Wikipedia search keywords: '{claim}'"
        search_query = model.generate_content(search_prompt).text.strip()
        
        logger.info(f"Searching Wikipedia for: {search_query}")
        search_results = wikipedia.search(search_query, results=1)
        
        if not search_results:
            return VerificationResult(status="unclear", claim=claim, evidence="No relevant Wikipedia page found.", source_url="", confidence="low")
        
        page = wikipedia.page(search_results[0], auto_suggest=False)
        
        # FIX 2: Use AI to actually READ the Wikipedia text and compare it to the claim
        verify_prompt = f"""
        Claim: {claim}
        Wikipedia Evidence: {page.summary[:1500]}
        
        Is the claim confirmed, refuted, or unclear based on the evidence?
        Provide a 1-sentence explanation.
        Format: [STATUS] | [EXPLANATION]
        """
        raw_res = model.generate_content(verify_prompt).text
        status_part = raw_res.split('|')[0].lower()
        explanation = raw_res.split('|')[-1].strip()

        status = "unclear"
        if "confirmed" in status_part: status = "confirmed"
        elif "refuted" in status_part: status = "refuted"

        return VerificationResult(
            status=status,
            claim=claim,
            evidence=explanation,
            source_url=page.url,
            confidence="high"
        )
    except Exception as e:
        return VerificationResult(status="unclear", claim=claim, evidence=f"Processing error: {str(e)}", source_url="", confidence="low")

@app.post("/extract-and-verify")
def extract_and_verify(request: TextRequest):
    claims = extract_claims(request.text)
    results = []
    for c in claims:
        res = verify_claim(ClaimRequest(claim=c))
        results.append(res.model_dump()) 
    return {"claims_found": len(claims), "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)