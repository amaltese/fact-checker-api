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
        model = genai.GenerativeModel('models/gemini-2.5-flash')
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
    model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
    
    try:
        # STEP 1: Context Recovery (Fixes "the city" / "it" issues)
        search_prompt = f"""
        Based on this claim, what is the best specific Wikipedia search query?
        If the claim uses pronouns like 'it' or 'the city', replace them with the correct subject.
        Return ONLY the search query string.
        Claim: {claim}
        """
        search_query_response = model.generate_content(search_prompt)
        search_query = search_query_response.text.strip()[:250]
        
        logger.info(f"Smart Search Query: {search_query}")
        search_results = wikipedia.search(search_query, results=1)
        
        if not search_results:
            return VerificationResult(status="unclear", claim=claim, evidence="No relevant Wikipedia articles found.", source_url="", confidence="low")

        page = wikipedia.page(search_results[0], auto_suggest=False)
        
        # STEP 2: Semantic Verification (Fixes Dutch East India Co vs West India Co)
        verify_prompt = f"""
        Claim: {claim}
        Wikipedia Evidence: {page.summary[:1500]}
        
        Is the claim CONFIRMED, REFUTED, or UNCLEAR based on the evidence? 
        Provide a 1-sentence explanation.
        Format: [STATUS] | [EXPLANATION]
        """
        
        # Add a 1-second delay to avoid Rate Limit (429) errors
        import time
        time.sleep(1)
        
        final_res = model.generate_content(verify_prompt).text.strip()
        
        # Ensure the response format is correct before splitting
        if "|" in final_res:
            status_part, explanation = final_res.split('|', 1)
        else:
            status_part, explanation = "unclear", final_res

        # The return MUST be inside the try block and indented correctly
        return VerificationResult(
            status=status_part.strip().lower(),
            claim=claim,
            evidence=explanation.strip(),
            source_url=page.url,
            confidence="high"
        )

    except Exception as e:
        logger.error(f"Error verifying claim '{claim}': {str(e)}")
        return VerificationResult(
            status="unclear", 
            claim=claim, 
            evidence=f"Error: {str(e)}", 
            source_url="", 
            confidence="low"
        )

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
