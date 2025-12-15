from fastapi import FastAPI, HTTPException
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

# Allow Custom GPT to call this API
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
    status: str  # "confirmed", "refuted", "unclear"
    claim: str
    evidence: str
    source_url: str
    confidence: str  # "high", "medium", "low"

def extract_claims(text: str) -> list:
    """
    Use Gemini to extract factual claims from text.
    Returns a list of claim strings.
    """
    if not GEMINI_API_KEY:
        logger.error("Cannot extract claims - GEMINI_API_KEY not configured")
        return []
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""Extract all factual claims from the following text. 
        Return ONLY a numbered list of discrete factual claims, one per line.
        Do not include opinions, interpretations, or analysis.
        Each claim should be a complete, verifiable statement.
        
        Text: {text}
        
        Claims:"""
        
        response = model.generate_content(prompt)
        claims_text = response.text
        
        # Parse the numbered list into individual claims
        claims = []
        for line in claims_text.split('\n'):
            line = line.strip()
            # Remove numbering (1., 2., etc.)
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove the number/bullet and clean up
                claim = line.lstrip('0123456789.-* ').strip()
                if claim:
                    claims.append(claim)
        
        logger.info(f"Extracted {len(claims)} claims from text")
        return claims
    
    except Exception as e:
        logger.error(f"Error extracting claims: {str(e)}")
        return []

@app.get("/")
def read_root():
    return {
        "message": "Wikipedia Fact Checker API",
        "endpoints": {
            "/verify-claim": "POST - Verify a factual claim against Wikipedia",
            "/extract-and-verify": "POST - Extract claims from text and verify each one"
        }
    }

@app.post("/verify-claim", response_model=VerificationResult)
def verify_claim(request: ClaimRequest):
    """
    Verify a factual claim against Wikipedia.
    Returns status, evidence, and source URL.
    """
    claim = request.claim.strip()
    
    if not claim:
        raise HTTPException(status_code=400, detail="Claim cannot be empty")
    
    logger.info(f"Verifying claim: {claim}")
    
    try:
        # Extract key terms from the claim for search
        # Simple approach: use first few significant words
        search_terms = claim.split()[:5]
        search_query = " ".join(search_terms)
        
        logger.info(f"Searching Wikipedia for: {search_query}")
        
        # Search Wikipedia
        search_results = wikipedia.search(search_query, results=3)
        
        if not search_results:
            return VerificationResult(
                status="unclear",
                claim=claim,
                evidence="No relevant Wikipedia articles found.",
                source_url="",
                confidence="low"
            )
        
        # Get the summary of the top result
        page_title = search_results[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        summary = page.summary
        
        # Simple verification logic
        # Check if key terms from claim appear in summary
        claim_lower = claim.lower()
        summary_lower = summary.lower()
        
        # Extract key terms (very basic - just words longer than 3 chars)
        claim_words = [w for w in claim_lower.split() if len(w) > 3]
        
        # Count how many key terms appear in summary
        matches = sum(1 for word in claim_words if word in summary_lower)
        match_ratio = matches / len(claim_words) if claim_words else 0
        
        # Determine status based on match ratio
        if match_ratio > 0.5:
            status = "confirmed"
            confidence = "medium" if match_ratio < 0.8 else "high"
            evidence = summary[:500] + "..." if len(summary) > 500 else summary
        elif match_ratio > 0.2:
            status = "unclear"
            confidence = "low"
            evidence = summary[:500] + "..." if len(summary) > 500 else summary
        else:
            status = "unclear"
            confidence = "low"
            evidence = f"Found article '{page_title}' but couldn't verify the specific claim. Summary: {summary[:300]}..."
        
        return VerificationResult(
            status=status,
            claim=claim,
            evidence=evidence,
            source_url=page.url,
            confidence=confidence
        )
    
    except wikipedia.exceptions.DisambiguationError as e:
        # Multiple possible articles - take the first option
        logger.info(f"Disambiguation needed, using: {e.options[0]}")
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            summary = page.summary[:500]
            return VerificationResult(
                status="unclear",
                claim=claim,
                evidence=f"Multiple articles found. Checking '{e.options[0]}': {summary}...",
                source_url=page.url,
                confidence="low"
            )
        except Exception:
            return VerificationResult(
                status="unclear",
                claim=claim,
                evidence="Multiple possible topics found. Please be more specific.",
                source_url="",
                confidence="low"
            )
    
    except wikipedia.exceptions.PageError:
        return VerificationResult(
            status="unclear",
            claim=claim,
            evidence="No Wikipedia page found for this topic.",
            source_url="",
            confidence="low"
        )
    
    except Exception as e:
        logger.error(f"Error verifying claim: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing claim: {str(e)}")

@app.post("/extract-and-verify")
def extract_and_verify(request: TextRequest):
    """
    Extract claims from text and verify each one against Wikipedia.
    Returns a list of verification results.
    """
    text = request.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    logger.info(f"Processing text of length {len(text)}")
    
    # Extract claims
    claims = extract_claims(text)
    
    if not claims:
        return {
            "original_text": text,
            "claims_found": 0,
            "results": [],
            "message": "No factual claims could be extracted from the text."
        }
    
    # Verify each claim
    results = []
    for claim in claims:
        try:
            claim_request = ClaimRequest(claim=claim)
            result = verify_claim(claim_request)
            results.append(result.dict())
        except Exception as e:
            logger.error(f"Error verifying claim '{claim}': {str(e)}")
            results.append({
                "status": "error",
                "claim": claim,
                "evidence": f"Error processing claim: {str(e)}",
                "source_url": "",
                "confidence": "low"
            })
    
    return {
        "original_text": text,
        "claims_found": len(claims),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)