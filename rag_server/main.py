from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests
import json

app = FastAPI()

class UploadRequest(BaseModel):
    full_text: str
    chunk_size: int

# 전역 변수로 모델과 클라이언트 초기화
embedding_model = None
chroma_client = None

@app.on_event("startup")
async def startup_event():
    global embedding_model, chroma_client
    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    chroma_client = chromadb.PersistentClient() # 그냥 클라이언트 - 저장 ㄴㄴ / 해당 코드 - 계속 이어짐

def split_text(full_text, chunk_size):
    encoder = tiktoken.encoding_for_model("gpt-5")
    total_encoding = encoder.encode(full_text)
    total_token_count = len(total_encoding)
    text_list = []
    for i in range(0, total_token_count, chunk_size):
        chunk = total_encoding[i: i+chunk_size]
        decoded = encoder.decode(chunk)
        text_list.append(decoded)

    return text_list

class MyEmbeddingFunction(EmbeddingFunction): #컬렉션이 벡터 디비 안에서 테이블 역할
    def __call__(self, input:Documents) -> Embeddings:
        return embedding_model.encode(input).tolist()
@app.post("/upload")
def upload(request: UploadRequest):
    global embedding_model, chroma_client

    chunk_list = split_text(request.full_text, request.chunk_size)
    embeddings = embedding_model.encode(chunk_list)
    print(embeddings)



    collection_name = 'github_collection'
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    github_collection = chroma_client.create_collection(name=collection_name,
                                                         embedding_function=MyEmbeddingFunction())
    id_list = []
    for index in range(len(chunk_list)):
        id_list.append(f'{index}')

    github_collection.add(documents=chunk_list, ids=id_list)

    return {"ok": True, "chunks": len(chunk_list)}

class QueryRequest(BaseModel):
    query: str
@app.post("/answer")
def ask(request: QueryRequest):
    global embedding_model, chroma_client
    collection_name = 'github_collection'
    github_collection = chroma_client.get_collection(name=collection_name, embedding_function=MyEmbeddingFunction())
    retrieved_doc = github_collection.query(query_texts=request.query, n_results=3)
    refer = retrieved_doc['documents'][0]

    url = "http://172.17.0.5:11434/api/generate"

    payload = {
        "model": "gemma3:1b",
        "prompt": f'''You are a GitHub repository assistant.

            Your role:
            - Answer questions using ONLY the provided Context.
            - The Context comes from a GitHub repository (code, README, directory structure, issues, etc).
            - Explain code and project structure clearly for developers.
            - If the answer is NOT found in the Context, say clearly that the information is not available in the repository.
            - Do NOT guess or hallucinate.
            - When helpful, mention file names or directories.
            
            Context:
            {refer}
            
            Question:
            {request.query}
            
            Answer in Korean.''',
        "stream": False
    }

    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()["response"]
