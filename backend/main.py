import os
import requests
from fastapi import FastAPI, Body, UploadFile, File
from pydantic import BaseModel
from starlette import status
from starlette.requests import Request
from starlette.responses import Response, RedirectResponse, JSONResponse
from fastapi import HTTPException

app = FastAPI()

RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://rag_server:8888")
OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL", "http://ollama:11434")