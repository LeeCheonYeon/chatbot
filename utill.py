import re
import logging
import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from ollama import Client
from typing import List, Dict, Any
from datetime import datetime, timedelta

import hashlib
import pymysql
import time
import html

"""설정 시작"""
EMBEDDING_URL = "http://localhost:7001/v1/embeddings"
EMBEDDING_MODEL = "BAAI/bge-m3"
QDRANT_URL = "http://localhost:7002"
RERANKER_URL = "http://localhost:8004"
COLLECTION_NAME = "test_cylee"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:7003"

client = QdrantClient(url=QDRANT_URL)
ollama_client = Client(host=OLLAMA_URL)

conn = pymysql.connect(
    host="192.168.0.97",
	port=3336,
    user="csia",
    password="Csia##2024",
    database="csia",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor
)

# 로그 설정
logging.basicConfig(level=logging.INFO)

# 한글, 영어, 숫자, 일본어, 한자만 허용
strip_pattern = re.compile(
    r"[^ a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥]"
)
strip_pattern2 = re.compile(
    r"[^ a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥.:/()_-]]"
)
"""설정 끝"""

# DB뷰 조회
def load_from_view():
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                PK_SEQ,
                TITLE,
                CONTENT,
                REG_DATE
            FROM vw_chatbot_contents
        """)
        return cur.fetchall()


#콜렉션 생성(QDRANT)
def create_collection(collection_nm:str = COLLECTION_NAME) -> None:
    if client.collection_exists(collection_nm): 
        print(f"🗑️ 기존 컬렉션 '{collection_nm}' 삭제 중...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    print(f"📦 컬렉션 '{collection_nm}' 생성 중...")
    vector_size = len(requests.post(
        EMBEDDING_URL,
        json={"input": "테스트", "model": EMBEDDING_MODEL},
        timeout=30
    ).json()["data"][0]["embedding"])
    
    if not client.collection_exists(collection_nm): 
        client.create_collection( collection_name=collection_nm, vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE), )

#콜렉션에 인덱스 생성(QDRANT)
def create_index(collection_nm:str = COLLECTION_NAME, filed_nm:str = "") -> None:
    if not client.collection_exists(collection_nm): 
        print(f"🗑️ 컬렉션 '{collection_nm}' 존재 하지 않음...")
    elif not filed_nm: 
        print("filed_nm 존재 하지 않음...")
    else:
        client.create_payload_index(
            collection_name=collection_nm,
            field_name=filed_nm,
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.MULTILINGUAL, # 한국어 형태소 대응을 위해 필수
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ) 
        )
        print(f"'{filed_nm}' 필드에 텍스트 인덱스 생성이 완료되었습니다.")
             
#임베딩 텍스트 > 벡터(TEI사용)
def get_embedding(text): 
    try:
        response = requests.post(EMBEDDING_URL, json={"model": EMBEDDING_MODEL, "input": text}, timeout=30 ) # 서버 응답에서 숫자 리스트(벡터)만 뽑아냅니다.
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        print(f"❌ 임베딩 실패: {e}")
        return None

# 문서 청크 임베딩 후 저장
def make_chunk_id(post_id, chunk_index, text):
    base = f"{post_id}_{chunk_index}_{text[:50]}"
    return int(hashlib.md5(base.encode()).hexdigest()[:12], 16)

#콜렉션에 데이터 저장(QDRANT)
def update_collection_data(collection_nm:str = COLLECTION_NAME, points:List[models.PointStruct] = []) ->None:
     if client.collection_exists(collection_nm) and points: 
        start_time = time.time()   # 시작 시간 기록
        BATCH_SIZE = 500  # 한 번에 업서트할 포인트 개수
        
        total_len = len(points)
        print(f"📦 총 {total_len}개의 데이터를 처리를 시작합니다. (배치 크기: {BATCH_SIZE})")
        for i in range(0, total_len, BATCH_SIZE):
            # 1. 500개씩 데이터 슬라이싱
            batch_points = points[i : i + BATCH_SIZE]
            client.upsert(collection_name=COLLECTION_NAME, points=batch_points)
            print(f"📦 {i + len(batch_points)} / {total_len} 완료...")
        print("✨ 모든 배치가 성공적으로 저장되었습니다.")
     else:
        if not points:
            print("⚠️ 업로드할 데이터(points)가 비어 있습니다.")
        else:
            print(f"❌ '{collection_nm}' 컬렉션이 존재하지 않습니다. 먼저 생성해 주세요.")

#콜렉션에 데이터 저장할때 해당 타입을 맞춰야되서, 변환하는 함수
def trans_list_to_pointStructList(documents:List = [], type:str = 'A') -> List[models.PointStruct]:
    points = []
    if type == 'A':
        for i, doc in enumerate(documents): 
            vector = get_embedding(doc) 
            # 실제 텍스트 문장을 함께 보관 
            points.append(models.PointStruct( id=i, vector=vector, payload={"text": doc} ))
    else:
        """형식에 맞춰서 구현해야함"""
        for i, doc in enumerate(documents):
            post_id = doc["PK_SEQ"]	
            title = doc["TITLE"] or ""
            content = doc["CONTENT"] or ""
            
            full_text = f"{title}\n{content}"
            full_text = html.unescape(full_text)
            bf_text = remove_tag_text(full_text)
            chunks = split_text(bf_text)
            for ii, chunk in enumerate(chunks):
                chunk_index = f"{i}_{ii}"
                id = make_chunk_id(post_id, chunk_index, chunk)
                vector = get_embedding(chunk)
                # 실제 텍스트 문장을 함께 보관 
                points.append(models.PointStruct( id=id, vector=vector, payload={ "post_id": post_id,
                    "chunk_index": ii,
                    "text": chunk,
                    "full_contents":full_text,
                    "updated_at": str(doc["REG_DATE"])} ))
    return points

#벡터로 검색
def search_collection_data(collection_nm: str = COLLECTION_NAME, query_vector:list = None, count:int=5):
    if client.collection_exists(collection_nm) and query_vector: 
        search_result = client.query_points(collection_name=collection_nm, query=query_vector, limit=count )
        return search_result
    else:
        if not query_vector:
            print("⚠️ 데이터(query_vector)가 비어 있습니다.")
        else:
            print(f"❌ '{collection_nm}' 컬렉션이 존재하지 않습니다. 먼저 생성해 주세요.")
        return []

#벡터 및 키워드로 검색
def search_collection_data_hybrid(collection_nm: str = COLLECTION_NAME, field_nm: str = "",vector_query_text:list = None, keyword_text:str = "", limit_count:int=5 ):
    if client.collection_exists(collection_nm) and vector_query_text and keyword_text: 
        # 2. 하이브리드 검색 실행
        search_result = client.query_points(
            collection_name=collection_nm,
            prefetch=[
                # (A) 벡터 검색: 의미적 유사성 기반 (2배수 추출)
                models.Prefetch(
                    query=vector_query_text,
                    limit=limit_count * 2,
                    score_threshold=0.5
                ),
                # (B) 키워드 검색: 특정 단어 포함 여부 기반 (2배수 추출)
                models.Prefetch(
                    # 점수 기준이 없어도 RRF가 순위를 매길 수 있도록 빈 벡터나 더미를 주지 않고,
                    # 필터만 적용된 Prefetch 구조를 사용합니다.
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=field_nm,
                                match=models.MatchText(text=keyword_text)
                            )
                        ]
                    ),
                    limit=limit_count * 2
                ),
            ],
            # (C) RRF 알고리즘으로 두 결과의 순위를 하나로 통합
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            score_threshold=0.5,
            limit=limit_count  # 전달할 최종 후보군 개수
        )
    return search_result

#리랭커로 순위 정렬
def get_rerank(query: str, documents: List[str]) -> List[Dict[str, Any]]:
    """
    리랭커 서버에 질의하여 문서들의 재순위화된 결과를 반환합니다.
    
    Args:
        query: 사용자 질문 문자열
        documents: 검색된 문서 텍스트 리스트
        
    Returns:
        재순위 점수와 인덱스 정보가 담긴 리스트 (예: [{'index': 0, 'score': 0.9}, ...])
    """
    response = requests.post( f"{RERANKER_URL}/rerank", json={"query": query, "texts": documents} )
    return response.json()

#리랭커로 순위 정렬 후 상위n개만 반환
def get_refined_context(query: str, documents: List[str], top_n: int = 5,min_score: float = 0.5) -> Dict[str, Any]:
    """
    리랭커 서버의 응답 형식([{"index": i, "score": s}, ...])을 그대로 유지하며
    상위 N개만 잘라서 반환합니다.
    """
    if not documents:
        return []

    # 1. 리랭커 서버에서 [{index: i, score: s}, ...] 형태를 가져옴
    reranked_data = get_rerank(query, documents)
   # 2. 필터링 및 텍스트 매칭
    refined_results = []
    for item in reranked_data:
        if item['score'] >= min_score:
            idx = item['index']
            # 리랭커가 준 인덱스를 사용하여 원본 documents에서 텍스트를 추출
            refined_results.append({
                "index": idx,
                "score": item['score'],
                "contexts": documents[idx]  # 👈 여기에 실제 내용을 넣어줍니다!
            })
            
        # top_n 개수만큼 찼으면 중단
        if len(refined_results) >= top_n:
            break
            
    return refined_results

def ask_ollama(context_text: str, user_query:str):
    # Ollama에게 보낼 최종 메시지 구조
    final_prompt = f"""[참고 자료]\n
                {context_text}\n
                [사용자 질문]\n
                {user_query}\n"""

    response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
        {
            "role": "system",
            "content": (
                "당신은 제공된 [참고 자료]에만 근거하여 답변하는 비즈니스 비서입니다.\n"
                "### 반드시 지켜야 할 출력 원칙 ###\n"
                "1. 자료에 답변 근거가 있다면: '핵심 결론'을 먼저 말하고 상세 내용을 경어체로 설명하세요.\n"
                "2. 자료에 답변 근거가 전혀 없다면: 군더더기 없이 '정보를 찾을 수 없습니다.' 딱 한 문장만 출력하세요.\n"
                "3. 질문의 의도를 파악할 수 없다면: '질문에 대해 모르겠습니다.' 딱 한 문장만 출력하세요.\n"
                "4. 절대 사견, 해설, '참고하여 작성했습니다' 등의 부연 설명을 하지 마세요.\n"
                "5. 모든 답변은 한국어로만 작성하세요."
        )
        },
        {
            'role': 'user',
            'content': final_prompt,
        },
       
    ],options={
            'temperature': 0,  # 일관된 답변을 위해 낮게 설정
            'num_ctx': 4096,
            'seed': 42,
        })
    return response['message']['content']

# 질문을 검색용으로 변경
def rewrite_question(question):
   
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
        {
            "role": "system",
            "content": (
                "### 역할 ###\n"
                "너는 사용자의 질문을 분석하여 '벡터 검색용 문장'과 '키워드 검색용 단어'로 변환하는 전문 쿼리 생성기이다.\n"
                 "지시사항: 지금부터 너는 인간의 말을 하는 AI가 아니라, 텍스트를 받으면'벡터 검색용 문장'과 '키워드 검색용 단어' 형태의 데이터만 뱉는 변환기이다. 인사말, 해설, 판단 근거를 출력하는 즉시 시스템 에러가 발생하므로 절대 출력하지 마라.\n"

                "### 규칙 ###\n"
                "1. 절대 사용자의 질문에 답하지 마라. (예: '~입니다', '~하세요' 금지)\n"
                "2. 반드시 질문에 없는 정보를 상상해서 추가하지 마라.\n"
                "3. 출력은 반드시 아래 형식을 지켜라.\n"

                "### 출력 형식 ###\n"
                "연관성: [질문과 연관됨/새로운 주제 중 선택]\n"
                "문장: [검색에 최적화된 완성형 문장] \n"
                "키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분]\n"

                "### 예시 ###\n"
                "입력: 거기 어떻게 가야 하지?\n"
                "문장: 해당 장소 방문 방법 및 대중교통 오시는 길 안내\n"
                "키워드: 방문 방법, 오시는 길, 교통편, 위치, 지도\n"
                "출력은 반드시 문장:[검색에 최적화된 완성형 문장]\n키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분] 만 나오게 해줘"
            )
        },
        {
            'role': 'user',
            'content':  f"입력: {question}",
        },
       
    ],options={
           'temperature': 0,      # 모델의 랜덤성을 완전히 제거 (가장 중요)
            'num_ctx': 4096,       # 컨텍스트 크기 (현재 질문 재작성에는 충분함)
            'seed': 42,            # 결과 재현을 위한 설정
            #'num_predict': 50,     # 모델이 내뱉는 글자 수를 제한 (사족 방지)
            'top_k': 1,  # 가장 확률이 높은 단어 1개만 고려
            'top_p': 1.0,
            'repeat_penalty': 1.0 # 반복 방지 로직이 개입하지 못하게 함
        })
    content = response['message']['content']
    # 문장과 키워드 분리
    vector_query = ""
    keyword_list = ""
    
    for line in content.split('\n'):
        if line.startswith("문장:"):
            vector_query = line.replace("문장:", "").strip()
        elif line.startswith("키워드:"):
            keyword_list = line.replace("키워드:", "").strip()
    return vector_query, keyword_list
        
               
#문장 전처리(html,한글태그 등 제거)
def remove_tag_text(raw_html: str) -> str:
    # HTML 분석
    soup = BeautifulSoup(raw_html, "lxml")
    
    # 본문과 상관없는 태그(스크립트, 스타일, 주석 등)는 아예 삭제
    for extra in soup(["script", "style", "header", "footer", "nav"]):
        extra.decompose()
        
    # 모든 링크(<a> 태그)를 찾아 "텍스트(URL)" 형태로 변환
    for a in soup.find_all('a'):
        href = a.get('href', '').strip()
        link_text = a.get_text().strip()
        
        if href and not href.startswith('#'): # 내부 이동 앵커 제외
            # 텍스트와 링크가 다르면 "텍스트(링크)"로, 같으면 하나만 표시
            new_content = f" {link_text}({href}) " if link_text != href else f" {href} "
            a.replace_with(new_content)
        else:
            # 주소가 없는 링크는 텍스트만 남김
            a.replace_with(link_text)
        
    # 텍스트 추출 (태그 간 간격을 주어 단어가 붙지 않게 함)
    text = soup.get_text(separator=" ")
    
    # HWP 제어 문자 등 비인쇄 문자 추가 제거
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text

def clean_for_reranker(text):
    # 1. 괄호 안의 URL 제거 (리랭커는 URL을 읽지 못함)
    text = re.sub(r'\(http[s]?://\S+\)', '', text)
    # 2. 일반 URL 제거
    text = re.sub(r'http[s]?://\S+', '', text)
    # 3. 이메일/전화번호 태그(tel:) 등 특수 태그 제거
    text = re.sub(r'\(tel:[^\)]+\)', '', text)
    # 4. 연속된 공백 하나로 통합
    text = " ".join(text.split())
    return text

# 문장 전처리(정규식으로 필요한 문자들 빼고 정리)
def clean_text(text: str) -> str:
    #허용된 문자만 남기고 공백 정리
    cleaned = strip_pattern2.sub("", text)
    return " ".join(cleaned.split())

# 문장 분리 , 문장클리닝 및 필터링, 문장병합
def split_text(text: str) -> List[str]:
    #문장 분리
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )

    chunks = []

    for chunk in text_splitter.split_text(text):
        #문장 클리닝 및 필터링
        cleaned = clean_text(chunk)

        logging.info(f"cleaned: {cleaned}")

        if len(cleaned) >= 30:
            #문장 병합
            chunks.append(cleaned)

    return chunks

# 1. 서버 메모리 저장소 (유저별 대화와 마지막 활동 시간 저장)
# { "user_id": { "messages": [...], "last_activity": datetime } }
memory_store = {}

def get_refined_context(user_id):
    now = datetime.now()
    
    # 해당 유저의 기록이 없으면 빈 리스트 반환
    if user_id not in memory_store:
        return []
    
    user_data = memory_store[user_id]
    
    # 2. 30분 제한 체크 (마지막 활동으로부터 30분이 지났으면 메모리 초기화)
    if now - user_data['last_activity'] > timedelta(minutes=30):
        memory_store[user_id] = {"messages": [], "last_activity": now}
        return []
    
    # 3. 3개 제한 (최신 질문-답변 세트 3개만 유지)
    # 질문/답변 쌍으로 저장되므로 리스트 길이는 최대 6개가 됩니다.
    return user_data['messages']

def update_memory(user_id, user_query, assistant_answer):
    now = datetime.now()
    
    if user_id not in memory_store:
        memory_store[user_id] = {"messages": [], "last_activity": now}
    
    # 메시지 추가
    memory_store[user_id]['messages'].append({"role": "user", "content": user_query})
    memory_store[user_id]['messages'].append({"role": "assistant", "content": assistant_answer})
    
    # 4. 뒤에서부터 3세트(6개 메시지)만 남기고 자르기
    if len(memory_store[user_id]['messages']) > 14:
        memory_store[user_id]['messages'] = memory_store[user_id]['messages'][-14:]
    
    # 마지막 활동 시간 갱신
    memory_store[user_id]['last_activity'] = now
    
# 질문을 검색용으로 변경하는데 기존 대화내용 포함하여
def rewrite_talk_question(user_id,question):
    try:
        history = get_refined_context(user_id)
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        print("###"*30)
        print(history_text)
    
        response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": (
                    "### 역할 ###\n"
                    "너는 기존 대화내용과 사용자의 질문을 분석하여 '벡터 검색용 문장'과 '키워드 검색용 단어'로 변환하는 전문 쿼리 생성기이다.\n"
                    "지시사항: 지금부터 너는 인간의 말을 하는 AI가 아니라, 텍스트를 받으면'벡터 검색용 문장'과 '키워드 검색용 단어' 형태의 데이터만 뱉는 변환기이다. 인사말, 해설, 판단 근거를 출력하는 즉시 시스템 에러가 발생하므로 절대 출력하지 마라.\n"
                    
                    "### 핵심 원칙 ###\n"
                    "사용자가 '거기', '그때', '그 내용' 등 '지시어'를 사용하지 않았다면, 이전 대화와는 완전히 다른 '새로운 주제'로 간주하고 마지막 질문만 독립적으로 변환한다. 이전 대화에 매몰되지 마라.\n"
                    "절대 대화 내용을 답변으로 내뱉지 마라. 오직 검색용 문장과 키워드만 생성하라.\n"
    
                    "### 규칙 ###\n"
                    "1. 절대 사용자의 질문에 답하지 마라. (예: '~입니다', '~하세요' 금지)\n"
                    "2. 결과물에 '관련이 없습니다', '무시합니다' 같은 해설을 절대 포함하지 마라.\n"
                    "3. 만약 마지막 질문이 이전 대화와 관계없는 '새로운 주제'라면, 이전 대화 내용을 무시하고 마지막 질문만으로 검색 데이터를 만드세요.\n"
                    "4. 질문 내에 '거기', '그분', '그때' 같은 지시어가 있는 경우에만 이전 대화를 참조하여 구체적인 명사로 치환하세요.\n"
                    "5. 질문이 완전히 독립적이라면(예: 갑자기 날씨를 묻거나 다른 기관을 묻는 경우), 이전 맥락을 섞지 마세요.\n"
                    "6. 질문에 없는 정보를 상상해서 추가하지 마라.\n"
                    "7. 출력은 반드시 아래 형식을 지켜라.\n"

                    "### 출력 형식 ###\n"
                    "연관성: [연관됨/새로운 주제 중 선택]\n"
                    "문장: [검색에 최적화된 완성형 문장] \n"
                    "키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분]\n"

                    "### 예시 ###\n"
                    "입력: 거기 어떻게 가야 하지?\n"
                    "문장: 해당 장소 방문 방법 및 대중교통 오시는 길 안내\n"
                    "키워드: 방문 방법, 오시는 길, 교통편, 위치, 지도\n"
                    "출력은 반드시 문장:[검색에 최적화된 완성형 문장]\n키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분] 만 나오게 해줘 \n"
                    
                    "### 예시 2 (맥락 무시) ###\n"
                    "이전 대화: 어린이집안전공제회 가입 방법은? / 연회비는 얼마야?\n"
                    "마지막 질문: 오늘 서울 날씨 어때?\n"
                    "출력:\n"
                    "문장: 오늘 서울 현재 기온 및 기상 상태 확인\n"
                    "키워드: 서울, 날씨, 기상, 기온\n"
                )
            },
            {
                'role': 'user',
                'content':  f"대화 내용: {history_text} 마지막 질문: {question}",
            },
        ],options={
            'temperature': 0,      # 모델의 랜덤성을 완전히 제거 (가장 중요)
            'num_ctx': 4096,       # 컨텍스트 크기 (현재 질문 재작성에는 충분함)
            'seed': 42,            # 결과 재현을 위한 설정
            #'num_predict': 50,     # 모델이 내뱉는 글자 수를 제한 (사족 방지)
            'top_k': 1,  # 가장 확률이 높은 단어 1개만 고려
            'top_p': 1.0,
            'repeat_penalty': 1.0 # 반복 방지 로직이 개입하지 못하게 함
        })
        content = response['message']['content']
        # 문장과 키워드 분리
        vector_query = ""
        keyword_list = ""
        
        for line in content.split('\n'):
            if line.startswith("문장:"):
                vector_query = line.replace("문장:", "").strip()
            elif line.startswith("키워드:"):
                keyword_list = line.replace("키워드:", "").strip()
    except Exception as e:
        print(f"❌ 질문 재생성 실패: {e}")
    return vector_query, keyword_list