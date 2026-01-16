from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import pymupdf4llm
import tiktoken
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests
import json

app = FastAPI()