import asyncio
import contextlib
import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import conversation, health
from app_logging.logger import get_logger

logger = get_logger(__name__)

async def self_ping_task():
    """
    Background task to ping the server's own health endpoint to prevent Render from sleeping.
    """
    # Wait 30 seconds after startup to allow the server to bind and start accepting requests
    await asyncio.sleep(30)
    
    external_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not external_url:
        logger.info("RENDER_EXTERNAL_URL not set. Skipping self-ping task.")
        return
        
    ping_url = f"{external_url.rstrip('/')}/health"
    logger.info(f"Starting self-ping background task targeting: {ping_url}")
    
    # 10 minutes (600 seconds) ping interval
    ping_interval = 600
    
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(ping_url, timeout=10.0)
                logger.info(f"Self-ping successful: {response.status_code}")
            except Exception as e:
                logger.error(f"Error during self-ping to {ping_url}: {e}")
            
            await asyncio.sleep(ping_interval)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the self-ping task in the background
    ping_task = asyncio.create_task(self_ping_task())
    yield
    # Shutdown: Cancel the self-ping task
    ping_task.cancel()
    try:
        await ping_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="CloudDash Support API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(conversation.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

