import os
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify API key for authentication"""
    expected_key = os.getenv("API_SECRET_KEY")
    
    if not expected_key:
        # If no API key is set, allow access (for development)
        return "development"
    
    if credentials.credentials != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return credentials.credentials
