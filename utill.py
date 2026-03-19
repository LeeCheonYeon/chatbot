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

import html2text

import hashlib
import pymysql
import time
import html
""" 설치
    pip install requests beautifulsoup4 qdrant-client langchain-text-splitters ollama pymysql
"""
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
    user="gmcc2021",
    password="Gmcc##2021",
    database="gmcc2021",
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor
)

conn2 = pymysql.connect(
    host="192.168.0.97",
	port=3336,
    user="gjeec2025",
    password="Gjeec##2025",
    database="gjeec2025",
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

def load_from_view2():
    with conn2.cursor() as cur:
        cur.execute("""
        SELECT CONCAT(SID,MID) AS PK_SEQ,
            HTML_NM         AS TITLE,
            HTML_CONTENT    AS CONTENT,
            MOD_DATE        AS REG_DATE
        FROM   vw_search_contents
        
        UNION ALL
        
        SELECT CONCAT(SID,'_',BID) AS PK_SEQ,
                TITLE,
                CONTENT AS CONTENT,
                REG_DATE
        FROM   vw_search_board
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
            client.upsert(collection_name=collection_nm, points=batch_points)
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
            
            full_text = f"{content}"
            full_text = html.unescape(full_text)
            bf_text = remove_tag_text(full_text)
            """
            chunks = split_text(title,bf_text)
            for ii, chunk in enumerate(chunks):
                chunk_index = f"{i}_{ii}"
                id = make_chunk_id(post_id, chunk_index, chunk)
                vector = get_embedding(chunk)
                #keyword = process_chunk(chunk)
                #print("===="*30)
                #print(keyword)
                #print("===="*30)
                #result_text = f"{keyword} {chunk}"
                # 실제 텍스트 문장을 함께 보관 
                points.append(models.PointStruct( id=id, vector=vector, payload={ "post_id": post_id,
                    "chunk_index": ii,
                    "text": chunk,
                    "full_contents":full_text,
                    "updated_at": str(doc["REG_DATE"])} ))
            """
            chunks = split_text2(title,bf_text)
            for ii, ch in enumerate(chunks):
                chunk = ch.get('chunk')
                rerank_chunk = ch.get('rerank_chunk')
                chunk_index = f"{i}_{ii}"
                id = make_chunk_id(post_id, chunk_index, chunk)
                vector = get_embedding(chunk)
                # 실제 텍스트 문장을 함께 보관 
                points.append(models.PointStruct( id=id, vector=vector, payload={ "post_id": post_id,
                    "chunk_index": ii,
                    "text": chunk,
                    "rerank_chunk":rerank_chunk,
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
        
        keyword_conditions = [
            models.FieldCondition(
                key=field_nm,
                match=models.MatchText(text=k.strip())
            ) for k in keyword_text.split(',') # 쉼표로 구분된 경우 쪼개기
        ]
        # 2. 하이브리드 검색 실행
        search_result = client.query_points(
            collection_name=collection_nm,
            prefetch=[
                # (A) 벡터 검색: 의미적 유사성 기반 (3배수 추출)
                models.Prefetch(
                    query=vector_query_text,
                    limit=limit_count * 3,
                    score_threshold=0.5
                ),
                # (B) 키워드 검색: 특정 단어 포함 여부 기반 (3배수 추출)
               models.Prefetch(
                    filter=models.Filter(
                        should=keyword_conditions # must 대신 should를 쓰면 검색 결과가 훨씬 풍성해집니다.
                    ),
                    limit=limit_count * 3
                )
            ],
            # (C) RRF 알고리즘으로 두 결과의 순위를 하나로 통합
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            score_threshold=0.5,
            limit=limit_count  # 전달할 최종 후보군 개수
        )
        # 2. 결과 리스트(points)만 뽑아서 'updated_at' 기준으로 정렬합니다.
        # raw_result.points가 실제 데이터 리스트입니다.
        sorted_points = sorted(
            search_result.points, 
            key=lambda x: x.payload.get('updated_at', 0) if x.payload else 0, 
            reverse=True
        )

        # 3. [핵심] 기존 query_points 결과와 동일한 구조로 다시 포장합니다.
        # 기존의 'search_result' 변수 구조와 똑같아집니다.
        search_res = models.QueryResponse(
            points=sorted_points[:limit_count] # 최신순 정렬된 상위 N개만 담음
        )

        # 이제 함수 밖에서 search_result.points를 호출해도 에러 없이 작동합니다!
        return search_res

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
def get_refined_context_rearrange(query: str, documents: List[str], top_n: int = 5,min_score: float = 0.5) -> Dict[str, Any]:
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
        print(f"점수 : {item['score']}")
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
    today = datetime.now().strftime("%Y년 %m월 %d일")
    final_prompt = f"""[참고 자료]\n
                {context_text}\n
                [사용자 질문]\n
                '{user_query}'에 대해 상세히 설명해줘."""
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": (
                    "당신은 제공된 [참고 자료]에만 근거하여 답변하는 '광주광역시도시공사' 비즈니스 비서입니다.\n\n"
                    "### 반드시 지켜야 할 출력 원칙 ###\n"
                    "- 절대 질문으로 답변하지 마세요.\n"
                    "- 자료에 답변 근거가 있다면: 상세 내용을 경어체로 설명하세요.\n"
                    "- 자료에 답변 근거가 전혀 없다면: 군더더기 없이 '정보를 찾을 수 없습니다.' 딱 한 문장만 출력하세요.\n"
                    "- 질문의 의도를 파악할 수 없다면: '질문에 대해 모르겠습니다.' 딱 한 문장만 출력하세요.\n"
                    "- 절대 사견, 해설, '참고하여 작성했습니다',''~을 통해 확인할 수 있습니다' 등의 부연 설명을 하지 마세요.\n"
                    "- 답변을 상상하거나 연관 없는 정보를 추가하지 마세요\n"
                    "- 모든 답변은 한국어를 기본으로 작성하세요.\n"
                    "- 분석 단계: [참고 자료] 내에 [사용자 질문]에 대한 직접적인 정보가 있는지 확인한다.\n"
                    "- 출력 단계 (성공): 정보가 있다면, 자료에 근거하여 상세히 답변한다. 반드시 경어체를 사용한다.\n"
                    "- 출력 단계 (성공): [참고 자료]를 분석해서 정확한 정보만 사용하여 답변한다.\n"
                    "- 출력 단계 (성공): 맞춤법, 한글 문법에 맞게 답변한다.\n"
                    "- 출력 단계 (실패): 자료에 직접적인 정보가 없거나, 질문이 자료와 관련 없다면 오직 '정보를 찾을 수 없습니다.' 한 문장만 출력한다.\n"
                    "- 절대 [참고 자료]를 짜집기 하지 마라\n"
                    "- 한문은 한글로 번역해서 답변해주세요.\n"
                    "- 자신[광주광역시도시공사 비즈니스 비서]을 소개하는 문장은 제외한다.\n"
                    "- 현재날짜를 확인하여 단어를 선택하라.\n"
                    "- 정확한 정보만 답변하세요.\n"
                    "- 인사와 본인이 누군지 말하지 마라.\n"
                    "- 문법에 맞게 답변하세요.\n"
                    "- 같은 내용을 반복해서 말하지 마세요.\n"
                     f"- 오늘 날짜는 {today}입니다. 모든 질문은 이 날짜를 기준으로 답변하세요.\n"
            )
            },
            {
                'role': 'user',
                'content': final_prompt,
            },
        
        ],options={
                'temperature': 0,  # 일관된 답변을 위해 낮게 설정
                'num_ctx': 8196,
                'seed': 42,
        },stream=True  # 스트리밍 활성화
    )
    for chunk in response:
        # 각 조각의 텍스트 내용만 추출해서 밖으로 던짐
        content = chunk['message']['content']
        yield content

def ask_ollama_follow(context_text: str, user_query:str, talk:str):
    today = datetime.now().strftime("%Y년 %m월 %d일")
    # Ollama에게 보낼 최종 메시지 구조
    final_prompt = f"""[대화 내용]\n
                {talk}\n
                [참고 자료]\n
                {context_text}\n
                [사용자 질문]\n
                '{user_query}"""
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": (
                    "당신은 제공된 [대화 내용] 과 [참고 자료]에만 근거하여 답변하는 '광주광역시도시공사' 비즈니스 비서입니다.\n\n"
                    "### 반드시 지켜야 할 출력 원칙 ###\n"
                    "- [사용자 질문]을 분석해서 [대화 내용]과 연관되지 않으면 [대화 내용]은 무시하세요.\n"
                    "- 절대 질문으로 답변하지 마세요.\n"
                    "- 자료에 답변 근거가 있다면: 상세 내용을 경어체로 설명하세요.\n"
                    "- 자료에 답변 근거가 전혀 없다면: 군더더기 없이 '정보를 찾을 수 없습니다.' 딱 한 문장만 출력하세요.\n"
                    "- 질문의 의도를 파악할 수 없다면: '질문에 대해 모르겠습니다.' 딱 한 문장만 출력하세요.\n"
                    "- 절대 사견, 해설, '참고하여 작성했습니다', '~을 통해 확인할 수 있습니다' 등의 부연 설명을 하지 마세요.\n"
                    "- 답변을 상상하거나 연관 없는 정보를 추가하지 마세요.\n"
                    "- 모든 답변은 한국어를 기본으로 작성하세요.\n"
                    "- 분석 단계: [대화 내용] 과 [참고 자료] 내에 [사용자 질문]에 대한 직접적인 정보가 있는지 확인한다.\n"
                    "- 출력 단계 (성공): 정보가 있다면, 자료에 근거하여 상세히 답변한다. 반드시 경어체를 사용한다.\n"
                    "- 출력 단계 (성공): [대화 내용]과 [참고 자료] 분석해서 정확한 정보만 사용하여 답변한다.\n"
                    "- 출력 단계 (성공): 맞춤법, 한글 문법에 맞게 답변한다.\n"
                    "- 출력 단계 (실패): 자료에 직접적인 정보가 없거나, 질문이 자료와 관련 없다면 오직 '정보를 찾을 수 없습니다.' 한 문장만 출력한다.\n"
                    "- 절대 [대화 내용]과 [참고 자료]를 짜집기 하지 마라\n"
                    "- 한문은 한글로 번역해서 답변해주세요.\n"
                    "- 자신[광주광역시도시공사 비즈니스 비서]을 소개하는 문장은 제외한다.\n"
                    "- 날짜를 확인하여 단어를 선택하라.\n"
                    "- 정확한 정보만 답변하세요.\n"
                    "- 인사와 본인이 누군지 말하지 마라.\n"
                    "- 문법에 맞게 답변하세요.\n"
                    "- 같은 내용을 반복해서 말하지 마세요.\n"
                    f"- 오늘 날짜는 {today}입니다. 모든 질문은 이 날짜를 기준으로 답변하세요.\n"
            )
            },
            {
                'role': 'user',
                'content': final_prompt,
            },
        
        ],options={
                'temperature': 0,  # 일관된 답변을 위해 낮게 설정
                'num_ctx': 8196,
                'seed': 42,
        },stream=True  # 스트리밍 활성화
    )
    for chunk in response:
        # 각 조각의 텍스트 내용만 추출해서 밖으로 던짐
        content = chunk['message']['content']
        yield content
        
# 질문을 검색용으로 변경
def rewrite_question(question):
   
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
        {
            "role": "system",
            "content": (
                "### 역할 ###\n"
                "너는 사용자의 질문[입력 데이터]을 분석하여 '벡터 검색용 문장'과 '키워드 검색용 단어'로 변환하는 '광주광역시도시공사' 전문 쿼리 생성기이다.\n"
                 "지시사항: 지금부터 너는 인간의 말을 하는 AI가 아니라, 텍스트를 받으면'벡터 검색용 문장'과 '키워드 검색용 단어' 형태의 데이터만 뱉는 변환기이다.\n"
                 "판단 근거를 출력하는 즉시 시스템 에러가 발생하므로 절대 출력하지 마라.\n"

                "### 절대 규칙 ###\n"
                "- 절대 사용자의 질문[입력 데이터]에 답변하지 마라.\n"
                "- 반드시 질문에 없는 정보를 상상해서 추가하지 마라.\n"
                "- [변환 결과]과 없을 경우 입력 데이터를 문장으로, 키워드는 입력 데이터에서 뽑아서 출력이 반드시 있게 하라.\n"
                "- 출력은 반드시 아래 형식을 지켜라.\n"
                "- 출력은 반드시 아래 형식외에 앞뒤로 특수문자는 절대로 붙이지 마라\n"
                "- 키워드에는 유의어도 포함시켜라\n"
                "- 키워드에는 유의어는 1차까지만 포함시켜라\n"
                
                "### 부정어 처리 규칙 ###\n"
                "- 사용자가 '~말고', '~제외하고', '~아닌' 등의 표현을 쓰면, 해당 단어는 [키워드]에서 완전히 제거하라.\n"
                "- 사용자가 강조한 '대체어'나 '목적어'를 중심으로 문장을 재구성하라.\n"
                "- 검색 쿼리에서 금지된 단어는 절대 포함하지 마라.\n"

                "### 출력 형식 ###\n"
                "연관성: [질문과 연관됨/새로운 주제 중 선택]\n"
                "문장: [검색에 최적화된 완성형 문장] \n"
                "키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분]\n"

                "### 예시 1 ###\n"
                "입력: 거기 어떻게 가야 하지?\n"
                "문장: 해당 장소 방문 방법 및 대중교통 오시는 길 안내\n"
                "키워드: 방문 방법, 오시는 길, 교통편, 위치, 지도\n"
                "출력은 반드시 문장:[검색에 최적화된 완성형 문장]\n키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분] 만 나오게 해줘\n"
                
                "### 예시 2 ###\n"
                "입력: [단어]는 뭔가요?\n"
                "문장: [단어]에 대한 정보\n"
                "키워드: [단어], [단어유의어]\n"
                "출력은 반드시 문장:[검색에 최적화된 완성형 문장]\n키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분] 만 나오게 해줘\n"
            )
        },
        {
            'role': 'user',
            'content':  f"### [입력 데이터] ###\n {question}\n\n### [변환 결과] ###",
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
    print(content)
    for line in content.split('\n'):
        line = line.strip() # 앞뒤 불필요한 공백 제거
        if not line: continue
        # '문장' 처리
        if "문장" in line:
            # 앞뒤의 모든 장식(특수문자/공백)을 날리고 알맹이(\1)만 남김
            vector_query = re.sub(r"^[^\w\s]*\s*문장\s*:\s*(.*?)\s*[^\w\s]*$", r"\1", line).strip()
        # '키워드' 처리
        elif "키워드" in line:
            keyword_list = re.sub(r"^[^\w\s]*\s*키워드\s*:\s*(.*?)\s*[^\w\s]*$", r"\1", line).strip()
    return vector_query, keyword_list
        
               
#문장 전처리(html,한글태그 등 제거)
def remove_tag_text(raw_html: str) -> str:
    raw_html = re.sub(r'<%.*?%>', '', raw_html)
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
    
    ## JSP 지시어 제거
    text = re.sub(r'<%.*?%>', '', text)
    
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
def split_text(title:str, text: str) -> List[str]:
    #문장 분리
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )

    chunks = []

    for chunk in text_splitter.split_text(text):
        #문장 클리닝 및 필터링
        chunk = f"{title} \n {chunk}"
        cleaned = clean_text(chunk)

        logging.info(f"cleaned: {cleaned}")

        if len(cleaned) >= 30:
            #문장 병합
            chunks.append(cleaned)

    return chunks

# 문장 분리 , 문장클리닝 및 필터링, 문장병합
def split_text2(title:str, text: str) -> List[str]:
    #문장 분리
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )

    # 2. 텍스트 분할
    docs = splitter.split_text(text)
    
    chunks = []
    
    for i in range(len(docs)):
        current = docs[i]
        
        # 앞뒤 컨텍스트 가져오기 (각 250자 내외로 제한하여 합계 900자 미만 유지)
        prev_context = docs[i-1][-250:] if i > 0 else ""
        next_context = docs[i+1][:250] if i < len(docs) - 1 else ""
        
        # 리랭커가 "가운데"가 핵심임을 알 수 있게 가공
        # 1,000자 이내로 밀도 높게 구성
        rerank_context = f"[PREV] {prev_context}\n[MAIN] {title} \n {current}\n[NEXT] {next_context}"
        
        chunks.append({
            "chunk": f"{title} \n {current}",      # 벡터 임베딩용
            "rerank_chunk": rerank_context,  # 리랭커 입력용
        })
    return chunks

# 1. 서버 메모리 저장소 (유저별 대화와 마지막 활동 시간 저장)
# { "user_id": { "messages": [...], "last_activity": datetime } }
memory_store = {}
"""
def get_refined_context(user_id):
    now = datetime.now()
    
    # 해당 유저의 기록이 없으면 빈 리스트 반환
    if user_id not in memory_store:
        return []
    
    user_data = memory_store[user_id]
    
    # 1. 30분 제한 체크
    if now - user_data['last_activity'] > timedelta(minutes=30):
        memory_store[user_id] = {"messages": [], "last_activity": now}
        return []
    
    messages = user_data['messages']
    if not messages:
        return []

    # 2. 메시지를 쌍(Pair)으로 묶어서 '최신성 가중치' 부여
    scored_pairs = []
    total_pairs = len(messages) // 2
    
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            # 현재 쌍이 몇 번째인지 (0부터 시작)
            pair_index = i // 2
            
            # 점수 계산: 뒤로 갈수록(최신일수록) 점수가 높음
            # 예: 3세트가 있다면 1/3, 2/3, 3/3 점 부여
            recency_score = (pair_index + 1) / total_pairs
            
            scored_pairs.append({
                "score": recency_score,
                "index": i,
                "items": [messages[i], messages[i+1]]
            })

    # 3. 점수가 높은(최신인) 순으로 상위 3세트 추출
    # 사실상 최신 3세트를 가져오는 것과 같지만, 나중에 점수 산정 방식을 
    # 바꾸더라도(예: 중요 키워드 포함 시 보너스 등) 구조가 유지됩니다.
    scored_pairs.sort(key=lambda x: x['score'], reverse=True)
    top_pairs = scored_pairs[:3]

    # 4. LLM 전달을 위해 다시 원래 시간 순서대로 정렬
    top_pairs.sort(key=lambda x: x['index'])
    # 5. 최종 리스트 생성
    refined_messages = []
    for pair in top_pairs:
        refined_messages.extend(pair['items'])
    return refined_messages
"""

def get_refined_context(user_id):
    now = datetime.now()
    if user_id not in memory_store:
        return []
    
    user_data = memory_store[user_id]
    
    # 30분 제한 체크
    if now - user_data['last_activity'] > timedelta(minutes=30):
        memory_store[user_id] = {"messages": [], "last_activity": now}
        return []
    
    messages = user_data['messages']
    if not messages:
        return []

    scored_pairs = []
    total_pairs = len(messages) // 2
    
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            pair_index = i // 2
            # 점수 계산 (최신일수록 1.0에 수렴)
            recency_score = round((pair_index + 1) / total_pairs, 2)
            
            scored_pairs.append({
                "score": recency_score,
                "index": i,
                "items": [messages[i], messages[i+1]]
            })

    # 최신 점수 상위 3세트 추출
    scored_pairs.sort(key=lambda x: x['score'], reverse=True)
    top_pairs = scored_pairs[:3]

    # 시간 순서로 재정렬
    top_pairs.sort(key=lambda x: x['index'])

    refined_messages = []
    for pair in top_pairs:
        score = pair['score']
        user_msg = pair['items'][0]['content']
        assistant_msg = pair['items'][1]['content']
        
        # 질문과 답변을 하나의 문자열로 결합
        combined_content = f"[점수: {score}] 질문: {user_msg} / 답변: {assistant_msg}"
        
        # 하나의 딕셔너리로 저장
        refined_messages.append({
            "context": combined_content
        })
            
    return refined_messages

def update_memory(user_id, user_query, assistant_answer):
    now = datetime.now()
    
    if user_id not in memory_store:
        memory_store[user_id] = {"messages": [], "last_activity": now}
    
    # 메시지 추가
    memory_store[user_id]['messages'].append({"role": "user", "content": user_query})
    memory_store[user_id]['messages'].append({"role": "assistant", "content": assistant_answer})
    
    # 4. 뒤에서부터 3세트(6개 메시지)만 남기고 자르기
    if len(memory_store[user_id]['messages']) > 6:
        memory_store[user_id]['messages'] = memory_store[user_id]['messages'][-6:]
    
    # 마지막 활동 시간 갱신
    memory_store[user_id]['last_activity'] = now
    
# 질문을 검색용으로 변경하는데 기존 대화내용 포함하여
def rewrite_talk_question(user_id,question):
    try:
        history = get_refined_context(user_id)
        history_text = "\n".join([item['context'] for item in history])
        response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
            {
                "role": "system",
                "content": (
                    "### 역할 ###\n"
                    "너는 사용자의 질문[마지막 질문]과 이전대화[대화 내용]을 분석하여 '벡터 검색용 문장'과 '키워드 검색용 단어'로 변환하는 '광주광역시도시공사' 전문 쿼리 생성기이다.\n"
                    "지시사항: 지금부터 너는 인간의 말을 하는 AI가 아니라, 텍스트를 받으면'벡터 검색용 문장'과 '키워드 검색용 단어' 형태의 데이터만 뱉는 변환기이다. \n"
                    "판단 근거를 출력하는 즉시 시스템 에러가 발생하므로 절대 출력하지 마라.\n"
                    
                    "### 핵심 원칙 ###\n"
                    "- 사용자의 질문[마지막 질문]을 기본으로 하라\n"
                    "- 대화 내용에 매몰되지 마라.\n"
                    "- 절대 대화 내용을 답변으로 내뱉지 마라. 오직 검색용 문장과 키워드만 생성하라.\n"
    
                    "### 절대 규칙 ###\n"
                    "- 절대 [마지막 질문]에 답변하지 마라.\n"
                    "- 결과물에 '관련이 없습니다', '무시합니다' 같은 해설을 절대 포함하지 마라.\n"
                    "- 대화 내용을 참조하여 구체적인 명사로 치환하세요.\n"
                    "- 질문에 없는 정보를 상상해서 추가하지 마라.\n"
                    "- 출력[변환 결과]은 반드시 아래 형식을 지켜라.\n"
                    "- 대화 내용을 그대로 질문으로 사용하지 말고 가공해서 사용하라.\n"
                    "- 출력[변환 결과]은 반드시 아래 형식외에 앞뒤로 특수문자는 절대로 붙이지 마라.\n"
                    "- 키워드에는 유의어도 포함시켜라\n"
                    "- 키워드에는 유의어는 1차까지만 포함시켜라\n"

                    "### 출력 형식 ###\n"
                    "연관성: [연관됨/새로운 주제 중 선택]\n"
                    "문장: [검색에 최적화된 완성형 문장] \n"
                    "키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분]\n"

                    "### 예시 ###\n"
                    "입력: 거기 어떻게 가야 하지?\n"
                    "문장: 해당 장소 방문 방법 및 대중교통 오시는 길 안내\n"
                    "키워드: 방문 방법, 오시는 길, 교통편, 위치, 지도\n"
                    "출력은 반드시 문장:[검색에 최적화된 완성형 문장]\n키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분] 만 나오게 해줘 \n"
                    
                    "### 예시 3 ###\n"
                    "입력: [단어]는 뭔가요?\n"
                    "문장: [단어]에 대한 정보\n"
                    "키워드: [단어], [단어유의어]\n"
                    "출력은 반드시 문장:[검색에 최적화된 완성형 문장]\n키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분] 만 나오게 해줘\n"
                )
            },
            {
                'role': 'user',
                'content':  f"대화 내용: {history_text} 마지막 질문: {question}\n\n### [변환 결과] ###",
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
        print(content)
        # 문장과 키워드 분리
        vector_query = ""
        keyword_list = ""
        
        for line in content.split('\n'):
            line = line.strip() # 앞뒤 불필요한 공백 제거
            if not line: continue
            # '문장' 처리
            if "문장" in line:
                # 앞뒤의 모든 장식(특수문자/공백)을 날리고 알맹이(\1)만 남김
                vector_query = re.sub(r"^[^\w\s]*\s*문장\s*:\s*(.*?)\s*[^\w\s]*$", r"\1", line).strip()
            # '키워드' 처리
            elif "키워드" in line:
                keyword_list = re.sub(r"^[^\w\s]*\s*키워드\s*:\s*(.*?)\s*[^\w\s]*$", r"\1", line).strip()
    except Exception as e:
        print(f"❌ 질문 재생성 실패: {e}")
    return vector_query, keyword_list

# 질문을 검색용으로 변경
def rewrite_question_keyword(question):
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
        {
            "role": "system",
            "content": (
                "### 역할 ###\n"
                 "- 너는 사용자의 질문[입력 데이터]을 분석하여 '키워드 검색용 단어'로 변환하는 '광주광역시도시공사'의 전문 쿼리 생성기이다.\n"
                 "- 지시사항: 지금부터 너는 인간의 말을 하는 AI가 아니라, 텍스트를 받으면 '키워드 검색용 단어' 형태의 데이터만 뱉는 변환기이다.\n"
                 "- 판단 근거를 출력하는 즉시 시스템 에러가 발생하므로 절대 출력하지 마라.\n"

                "### 절대 규칙 ###\n"
                "- 절대 사용자의 질문[입력 데이터]에 답변하지 마라.\n"
                "- 반드시 질문에 없는 정보를 상상해서 추가하지 마라.\n"
                "- 출력은 반드시 아래 형식을 지켜라.\n"
                "- 출력은 반드시 아래 형식외에 앞뒤로 특수문자는 절대로 붙이지 마라.\n"
                "- 키워드에는 유의어도 포함시켜라.\n"
                "- 키워드에는 유의어는 1차까지만 포함시켜라.\n"
                "- 띄어쓰기가 있으면 띄어쓰기 기준으로 단어라고 생각하라."
                
                "### 부정어 처리 규칙 ###\n"
                "- 사용자가 '~말고', '~제외하고', '~아닌' 등의 표현을 쓰면, 해당 단어는 [키워드]에서 완전히 제거하라.\n"

                "### 출력 형식 ###\n"
                "연관성: [질문과 연관됨/새로운 주제 중 선택]\n"
                "키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분]\n"

                "### 예시 1 ###\n"
                "입력: 거기 어떻게 가야 하지?\n"
                "키워드: 방문 방법, 오시는 길, 교통편, 위치, 지도\n"
                "출력은 반드시\n키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분] 만 나오게 해줘\n"
                
                "### 예시 2 ###\n"
                "입력: [단어]는 뭔가요?\n"
                "키워드: [단어], [단어유의어]\n"
                "출력은 반드시 키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분] 만 나오게 해줘\n"
                
                "### 예시 3 ###\n"
                "입력: [단어1] [단어2] 어떻게 되나요?\n"
                "키워드: [단어1], [단어2], [단어1유의어],[단어2유의어]\n"
                "출력은 반드시 키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분] 만 나오게 해줘\n"
            )
        },
        {
            'role': 'user',
            'content':  f"### [입력 데이터] ###\n {question}\n\n### [변환 결과] ###",
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
    
    keyword_list = ""
    print(content)
    for line in content.split('\n'):
        if "키워드" in line:
            keyword_list = re.sub(r"^[^\w\s]*\s*키워드\s*:\s*(.*?)\s*[^\w\s]*$", r"\1", line).strip()
    return keyword_list


def check_is_follow_up(user_input):
    # 1. 재질문 핵심 키워드 패턴 (지시어, 이유, 대조 등)
    follow_up_patterns = [
        r"(그거|그것|그게|그건|거기|그때|이중|저번|방금|이전|아까|다시|그런데|전에|그곳|저곳|이곳|거기서|지금)", # 지시어
        r"^(왜|어째서|이유가|근거가|진짜|정말|확실해|맞아|근데|그럼)",     # 이유 및 확인 (문장 시작)
        r"(더|추가로|자세히|상세히|구체적으로|해당|말한|그사람|이사람)",           # 상세 설명 요청
        r"(다른|대신|말고|아니면|차이|비교)"              # 대안 및 비교
    ]
    
    # 2. 입력값 전처리 (공백 제거 등)
    text = user_input.strip()
    
    # 3. 판별 조건 설정
    is_keyword_match = any(re.search(pattern, text) for pattern in follow_up_patterns)
    
    # 최종 판별: 키워드가 매칭되거나, 문장이 아주 짧으면서 특정 조사로 끝날 때
    if is_keyword_match:
        return True
    return False



# 질문을 검색용으로 변경
def rewrite_question_keyword2(question):
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
        {
            "role": "system",
            "content": (
                "### 역할 ###\n"
                 "- 너는 사용자의 질문[입력 데이터]을 분석하여 '키워드 검색용 단어'로 변환하는 전문 쿼리 생성기이다.\n"
                 "- 지시사항: 지금부터 너는 인간의 말을 하는 AI가 아니라, 텍스트를 받으면 '키워드 검색용 단어' 형태의 데이터만 뱉는 변환기이다.\n"
                 "- 판단 근거를 출력하는 즉시 시스템 에러가 발생하므로 절대 출력하지 마라.\n"

                "### 절대 규칙 ###\n"
                "- 절대 사용자의 질문[입력 데이터]에 답변하지 마라.\n"
                "- 반드시 질문에 없는 정보를 상상해서 추가하지 마라.\n"
                "- 출력은 반드시 아래 형식을 지켜라.\n"
                "- 출력은 반드시 아래 형식외에 앞뒤로 특수문자는 절대로 붙이지 마라\n"
                "- 키워드에는 유의어도 포함시켜라\n"
                "- 키워드에는 유의어는 1차까지만 포함시켜라\n"
                
                "### 부정어 처리 규칙 ###\n"
                "- 사용자가 '~말고', '~제외하고', '~아닌' 등의 표현을 쓰면, 해당 단어는 [키워드]에서 완전히 제거하라.\n"

                "### 출력 형식 ###\n"
                "연관성: [질문과 연관됨/새로운 주제 중 선택]\n"
                "키워드: [검색 필터로 사용할 핵심 명사들 or 로 구분]\n"

                "### 예시 1 ###\n"
                "입력: 거기 어떻게 가야 하지?\n"
                "키워드: 방문 방법 or 오시는 길 or 교통편 or 위치 or 지도\n"
                "출력은 반드시\n키워드: [검색 필터로 사용할 핵심 명사들 or 로 구분] 만 나오게 해줘\n"
                
                "### 예시 2 ###\n"
                "입력: [단어]는 뭔가요?\n"
                "키워드: [단어]or [단어유의어]\n"
                "출력은 반드시 키워드: [검색 필터로 사용할 핵심 명사들, 쉼표로 구분] 만 나오게 해줘\n"
            )
        },
        {
            'role': 'user',
            'content':  f"### [입력 데이터] ###\n {question}\n\n### [변환 결과] ###",
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
    
    keyword_list = ""
    print(content)
    for line in content.split('\n'):
        if "키워드" in line:
            keyword_list = re.sub(r"^[^\w\s]*\s*키워드\s*:\s*(.*?)\s*[^\w\s]*$", r"\1", line).strip()
    return keyword_list

def rewrite_query_with_history(user_id, current_query):
    # chat_history는 이전 대화 리스트
    history = get_refined_context(user_id)
    history_text = "\n".join([item['context'] for item in history])
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
        {
            "role": "system",
            "content": (
                "### 역할 ###\n"
                "- 너는 사용자의 질문[마지막 질문]과 이전대화[대화 내용]을 분석하여 '단독으로 검색 가능한 질문'으로 변환하는 '광주광역시도시공사' 전문 쿼리 생성기이다.\n\n"
        
                "### 절대 규칙 ###\n"
                "- 절대 [마지막 질문]에 답변하지 마라.\n"
                "- 반드시 질문에 없는 정보를 상상해서 추가하지 마라.\n"
                "- [이전대화]의 점수는 최신 일수록 높습니다 점수가 높을수록 더 많은 가중치를 주세요.\n"
                "- 출력은 앞뒤로 특수문자는 절대로 붙이지 마라\n"
                "- [이전대화]와 관련없으면 [마지막 질문]을 출력하세요.\n"
                "- 출력은 반드시 아래 형식을 지켜라.\n"
                "- 질문만 출력하라.\n"
                "- [마지막 질문]이 제일 중요하다.\n"
                "- 한 문장으로 질문을 만들어라.\n"
                
                "### 출력 형식 ###\n"
                "[만든 질문]"
            )
        },
        {
            'role': 'user',
            'content':  f" 이전대화: {history_text} \n\n 마지막 질문: {current_query}\n\n ### [변환 결과] ###",
        },
        ],options={'temperature': 0, 'seed': 42})
    return response['message']['content']

def process_chunk(chunk, org_name="광주광역시도시공사"):
    prompt = f"""
    문서 조각을 보고 검색 성능을 높이기 위한 요약과 핵심 키워드 3개만 뽑으세요.
    주어가 없다면 반드시 '{org_name}'을 포함하세요.
    출력 형식: [요약], [키워드1, 키워드2, 키워드3] (설명 없이 단어만)
    문서: {chunk[:500]} # 토큰 절약을 위해 앞부분만 전달
    """
    try:
        response = ollama_client.generate(model=OLLAMA_MODEL, prompt=prompt, options={'temperature': 0, 'seed': 42})
        keywords = response['response'].strip()
        return keywords
    except Exception as e:
        print(f"Ollama 에러: {e}")
        return f"[{org_name}]" # 에러 시 기본 기관명이라도 반환

def normalize_table_merged_cells(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    tables = soup.find_all('table')
    
    for table in tables:
        rows = table.find_all('tr')
        if not rows: continue
        
        # 1. 실제 표의 최대 행/열 크기 파악
        num_rows = len(rows)
        num_cols = 0
        for row in rows:
            cells = row.find_all(['td', 'th'])
            count = sum(int(c.get('colspan', 1)) for c in cells)
            num_cols = max(num_cols, count)
            
        # 2. 가상의 그리드(2차원 배열) 생성
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        
        # 3. 그리드를 돌며 병합된 셀 내용 복제
        for r_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            c_idx = 0
            for cell in cells:
                # 이미 rowspan 등에 의해 채워진 칸 건너뛰기
                while c_idx < num_cols and grid[r_idx][c_idx] is not None:
                    c_idx += 1
                
                if c_idx >= num_cols: break
                
                content = cell.get_text(strip=True)
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                
                # 병합된 범위만큼 내용 채우기 (정규화)
                for r_offset in range(rowspan):
                    for c_offset in range(colspan):
                        if r_idx + r_offset < num_rows and c_idx + c_offset < num_cols:
                            grid[r_idx + r_offset][c_idx + c_offset] = content
                c_idx += colspan
        
        # 4. 기존 HTML table 내부를 정규화된 데이터로 재구성
        table.clear() # 기존 내용 비우기
        for r_data in grid:
            new_tr = soup.new_tag('tr')
            for c_data in r_data:
                new_td = soup.new_tag('td')
                new_td.string = c_data if c_data else ""
                new_tr.append(new_td)
            table.append(new_tr)
            
    return str(soup)

def process_to_markdown(raw_html):
    # [1] JSP 지시어 및 주석 먼저 제거 (HTML 파싱 방해 요소 제거)
    raw_html = re.sub(r'<%.*?%>', '', raw_html, flags=re.DOTALL)
    raw_html = re.sub(r'', '', raw_html, flags=re.DOTALL)
    
    # [2] BeautifulSoup으로 노이즈 태그 제거
    soup = BeautifulSoup(raw_html, "lxml")
    for extra in soup(["script", "style", "header", "footer", "nav", "iframe"]):
        extra.decompose()

    # [3] 마크다운 변환 (soup을 다시 문자열로 바꿔서 전달)
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0  # 줄바꿈 방지
    
    # soup.decode()를 써서 정제된 HTML 상태로 변환기에 넘깁니다.
    markdown_text = h.handle(str(soup))

    # [4] 최종 마크다운 텍스트에서 비인쇄 문자 및 불필요한 공백 제거
    # HWP 제어 문자 등 비인쇄 문자 제거
    markdown_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', markdown_text)
    # 너무 많은 줄바꿈 정리 (3개 이상을 2개로)
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    
    return markdown_text.strip()