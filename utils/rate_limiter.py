import time
from collections import defaultdict, deque
from fastapi import HTTPException

class RateLimiter:
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(deque)
    
    async def check_rate_limit(self, client_id: str):
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests outside time window
        while client_requests and client_requests[0] < current_time - self.time_window:
            client_requests.popleft()
        
        # Check if limit exceeded
        if len(client_requests) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.time_window} seconds"
            )
        
        # Record new request
        client_requests.append(current_time)
