from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests
import json
from typing import List
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 상수 정의
EMBEDDING_MODEL_NAME = 'jhgan/ko-sroberta-multitask'
TOKENIZER_MODEL = "gpt-5"
COLLECTION_NAME = 'github_collection'
OLLAMA_URL = "http://172.17.0.5:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"


# Pydantic 모델
class UploadRequest(BaseModel):
    full_text: str = Field(..., min_length=1)
    chunk_size: int = Field(..., gt=0)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class UploadResponse(BaseModel):
    ok: bool
    chunks: int


# 전역 변수
embedding_model: SentenceTransformer = None
chroma_client: chromadb.PersistentClient = None


# 임베딩 함수 클래스
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return embedding_model.encode(input).tolist()


# 유틸리티 함수
def split_text(full_text: str, chunk_size: int) -> List[str]:
    """텍스트를 청크 단위로 분할"""
    encoder = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    total_encoding = encoder.encode(full_text)
    total_token_count = len(total_encoding)

    text_list = []
    for i in range(0, total_token_count, chunk_size):
        chunk = total_encoding[i:i + chunk_size]
        decoded = encoder.decode(chunk)
        text_list.append(decoded)

    return text_list


def initialize_collection() -> chromadb.Collection:
    """컬렉션 초기화 (기존 삭제 후 생성)"""
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"기존 컬렉션 '{COLLECTION_NAME}' 삭제 완료")
    except Exception as e:
        logger.info(f"기존 컬렉션 없음: {e}")

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=MyEmbeddingFunction()
    )
    logger.info(f"새 컬렉션 '{COLLECTION_NAME}' 생성 완료")
    return collection


def query_ollama(prompt: str) -> str:
    """Ollama API 호출"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(OLLAMA_URL, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API 호출 실패: {e}")
        raise HTTPException(status_code=500, detail="AI 모델 응답 실패")


# FastAPI 이벤트
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 모델 및 클라이언트 초기화"""
    global embedding_model, chroma_client

    logger.info("임베딩 모델 로드 중...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info("ChromaDB 클라이언트 초기화 중...")
    chroma_client = chromadb.PersistentClient()

    logger.info("초기화 완료")


# API 엔드포인트
@app.post("/upload", response_model=UploadResponse)
def upload(request: UploadRequest):
    """텍스트를 청크로 분할하여 벡터 DB에 저장"""
    try:
        # 텍스트 분할
        chunk_list = split_text(request.full_text, request.chunk_size)
        logger.info(f"텍스트 분할 완료: {len(chunk_list)}개 청크")

        # 임베딩 생성 (로그 확인용)
        embeddings = embedding_model.encode(chunk_list)
        logger.info(f"임베딩 생성 완료: shape={embeddings.shape}")

        # 컬렉션 초기화 및 데이터 추가
        collection = initialize_collection()
        id_list = [str(i) for i in range(len(chunk_list))]
        collection.add(documents=chunk_list, ids=id_list)

        logger.info(f"벡터 DB 저장 완료: {len(chunk_list)}개 문서")
        return UploadResponse(ok=True, chunks=len(chunk_list))

    except Exception as e:
        logger.error(f"업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"업로드 처리 실패: {str(e)}")


@app.post("/answer")
def ask(request: QueryRequest):
    """질문에 대한 답변 생성"""
    try:
        # 컬렉션 가져오기
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=MyEmbeddingFunction()
        )

        # 유사 문서 검색
        retrieved_doc = collection.query(query_texts=request.query, n_results=3)
        refer = retrieved_doc['documents'][0]
        logger.info(f"검색된 문서 수: {len(refer)}")

        # 프롬프트 생성
        prompt = f'''You are a GitHub repository assistant.

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
            
            Answer in Korean.'''

        # AI 응답 생성
        answer = query_ollama(prompt)
        return {"answer": answer}

    except Exception as e:
        logger.error(f"답변 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {str(e)}")


@app.get("/health")
def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "ollama_model": OLLAMA_MODEL
    }