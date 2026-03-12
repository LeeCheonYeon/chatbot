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
        """
        if check_follow_up and old_talk :
            #ollama를 통해서 질문을 검색용 문장 및 키워드로 변경
            vector_query, keyword_list = utill.rewrite_talk_question(USER_ID,user_query)
        else:
            keyword_list = utill.rewrite_question_keyword(user_query)
            vector_query = user_query
        """
        keyword_list = utill.rewrite_question_keyword(user_query)
        vector_query = user_query
        print("="*60)
        print(keyword_list)
        if keyword_list:
            try:
                # 2. Qdrant 벡터 검색 (1단계: 관련 문서 후보 5개 추출)
                print("🔍 관련 자료를 검색하고 있습니다...")
                user_query_emb = utill.get_embedding(vector_query)
                initial_docs = utill.search_collection_data_hybrid("test_cylee","full_contents",user_query_emb,keyword_list, 5)
                if not initial_docs:
                    print("⚠️ 검색된 기본 자료가 없습니다.")
                    continue
                for point in initial_docs.points:
                    print(f"ID: {point.id}, Score: {point.score}")
                print(f"{len(initial_docs.points)} 개의 참고자료")

                refined_docs = [point.payload.get("text", "") for point in initial_docs.points]
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"*3)
                print(refined_docs)
                # 3. 리랭커 필터링 (2단계: 점수 0.5 이상, 상위 3개 정밀 선별)
                # 이제 이 함수가 텍스트까지 포함된 리스트를 반환합니다.
                print("🎯 자료의 정확도를 분석 중입니다...")
                if check_follow_up and old_talk : 
                   refined_query = utill.rewrite_query_with_history(USER_ID,user_query)
                else :
                   refined_query =  f"{vector_query}({keyword_list})"
                print(refined_query)
                refined_data = utill.get_refined_context_rearrange(
                    query=refined_query, 
                    documents=refined_docs, 
                    top_n=2, 
                    min_score=0.05
                )

                # 4. 컨텍스트 구성
                if not refined_data and not check_follow_up and  not old_talk:
                    # 검색은 됐으나 점수가 너무 낮아 신뢰할 수 없는 경우
                    context_text = "" 
                    print("💡 참고할 만한 충분한 점수의 자료를 찾지 못했습니다.")
                    continue
                else:
                    # 검색된 텍스트들을 하나로 합침
                    context_text = "\n".join([initial_docs.points[item['index']].payload['full_contents'] for item in refined_data])
                    print(f"✅ {len(refined_data)}개의 핵심 근거를 찾았습니다.")
                    context_text = utill.remove_tag_text(context_text)
                    # (선택 사항) 디버깅용 점수 출력
                    if refined_data:
                        scores = [f"{item['score']:.2f}" for item in refined_data]
                        print(f"참고 자료 신뢰도: {', '.join(scores)}")
                    
                if context_text or (check_follow_up and old_talk):
                    # 5. Ollama 답변 생성 (최종 단계)
                    print("✍️ 답변을 생성하는 중입니다. 잠시만 기다려 주세요...\n")
                    print(f"{len(context_text)} 길이")
                    if check_follow_up and old_talk :
                        answer = utill.ask_ollama_follow(context_text,vector_query,old_talk)
                    else:
                        answer = utill.ask_ollama(context_text,vector_query)
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