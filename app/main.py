import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langfuse import get_client

from app.chat import router as chat_router
from app.database import close_mongo_connection, connect_to_mongo
from app.routes import router as routes_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()
    get_client().flush()


app = FastAPI(title="Interior Design Chat", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/viewer")
async def viewer():
    return FileResponse("app/static/viewer.html")


app.include_router(chat_router)
app.include_router(routes_router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
