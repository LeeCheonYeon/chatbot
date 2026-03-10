import requests
from urllib.parse import quote
import utill
import traceback

#나중에 세션아이디로 받아서 처리
USER_ID = 'test'

def run_consulting_system():
    """메인 루프"""
    
    print("="*60)
    print("🤖 [챗봇]이 가동되었습니다.")
    print("질문을 입력하세요 (종료하려면 '나가기' 입력)")
    print("="*60)

    while True:
        # 1. 사용자 질문 입력
        user_query = input("\n❓ 질문: ").strip()
        
        if user_query in ['나가기', 'exit', 'quit']:
            print("👋 시스템을 종료합니다. 이용해 주셔서 감사합니다.")
            break
        
        if not user_query:
            continue
        
        old_talk = utill.get_refined_context(USER_ID)
        print(old_talk)
        check_follow_up = utill.check_is_follow_up(user_query)
        user_query = utill.remove_tag_text(user_query)
        vector_query = ""
        if check_follow_up and old_talk :
            #ollama를 통해서 질문을 검색용 문장 및 키워드로 변경
            vector_query, keyword_list = utill.rewrite_talk_question(USER_ID,user_query)
        else:
            keyword_list = utill.rewrite_question_keyword2(user_query)
            vector_query = user_query
        print("="*60)
        print(keyword_list)
        if keyword_list:
            try:
               
               # 1. API 엔드포인트(URL) 설정
                url = "http://211.228.51.207:9503/findeep/search"
                """
                http://211.228.51.207:9503/findeep/search?indexName=&keyword=%ED%85%8C%EC%8A%A4%ED%8A%B8&page=1&select=&sort=sid:asc,created:desc&dateField=created&sdate=&edate=&pageCnt=20&sessionId=8860BAA33C24236EA3A61E6464168320&includeField=site_nm,link_url,menu_name_full,title,_content,reg_date,staff_nm,stat,stat_nm,tel,business,org_nm,img_url,file_view_url,file_url,bid,list_no,file_name_org,menu_type,path,highlight,filename,b_type,address,data04,reg_date,sid,file_url,alt,_score,home&excludeField=null&divVal=a1&divKey=sid
                """
                
                keyword = (vector_query+ ' or '+keyword_list) if vector_query else keyword_list
                encoded_text = quote(keyword_list)
                # 2. 보낼 데이터 (파이썬 딕셔너리 형태)
                payload = {
                "dateField": "created",
                "divKey": "sid",
                "divVal": "a1",
                "edate": "",
                "excludeField": "",
                "includeField": "title,html_content,content,reg_date",
                "indexName": "vw_search_contents,vw_search_board",
                "keyword": encoded_text,
                "page": "1",
                "pageCnt": "20",
                "sdate": "",
                "select": "",
                "sessionId": "8860BAA33C24236EA3A61E6464168320",
                "sort": "sid:asc,created:desc"
                }

                # 2. 헤더 설정 (자바의 setRequestProperty 대응)
                headers = {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "charset": "UTF-8"
                }

                # 3. POST 요청 보내기 (json= 파라미터 활용)
                response = requests.post(
                        url, 
                        data=payload, 
                        headers=headers, 
                        timeout=5 
                    )
                refined_docs = []
                # 4. 결과 확인
                if response.status_code == 200:
                    print("성공적으로 전송되었습니다!")
                    #print("서버 응답:", response.json())
                    result = response.json()
                    print(result)
                    indexs = result['indexes']
                    content = indexs['vw_search_contents']
                    board = indexs['vw_search_board']
                    c_cnt = content['totalCount']
                    b_cnt = board['totalCount']
                    c_docs = []
                    b_docs = []
                    if(c_cnt > 0):
                        c_docs = [point.get("html_content", "") for point in content['dataset']] 
                    
                    if(b_cnt > 0):
                        b_docs = [point.get("content", "") for point in board['dataset']] 
                        
                    refined_docs = c_docs + b_docs
                    print(refined_docs)
                else:
                    print(f"실패 코드: {response.status_code}")

                # 3. 리랭커 필터링 (2단계: 점수 0.5 이상, 상위 3개 정밀 선별)
                # 이제 이 함수가 텍스트까지 포함된 리스트를 반환합니다.
                print("🎯 자료의 정확도를 분석 중입니다...")
                
                    
                if refined_docs:
                    # 5. Ollama 답변 생성 (최종 단계)
                    print("✍️ 답변을 생성하는 중입니다. 잠시만 기다려 주세요...\n")
                    print(f"{len(refined_docs)} 길이")
                    if check_follow_up and old_talk :
                        answer = utill.ask_ollama_follow(refined_docs,vector_query,old_talk)
                    else:
                        answer = utill.ask_ollama(refined_docs,vector_query)
                    full_text = "" # 전체 답변
                    for chunk in answer:
                        print(chunk, end='', flush=True) # 사용자에게 실시간 출력
                        full_text += chunk
                    utill.update_memory(USER_ID, vector_query, full_text)
                    print()
                else:
                    for chunk in '질문에 대한 답변을 찾지 못하였습니다.':
                        print(chunk, end='', flush=True) # 사용자에게 실시간 출력
                    print()
            except Exception as e:
                traceback_message = traceback.format_exc()
                print(traceback_message)
                print(f"❌ 오류가 발생했습니다: {e}")
        else:
            for chunk in '질문에 대한 답변을 찾지 못하였습니다.':
                print(chunk, end='', flush=True) # 사용자에게 실시간 출력
            print()

if __name__ == "__main__":
    run_consulting_system()
    
    
    
    

