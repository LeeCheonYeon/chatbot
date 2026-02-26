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
        
        
        old_query = utill.get_refined_context(USER_ID)
        user_query = utill.remove_tag_text(user_query)
        
        if not old_query:
            #ollama를 통해서 질문을 검색용 문장 및 키워드로 변경
            vector_query, keyword_list = utill.rewrite_question(user_query)
        else:
             #ollama를 통해서 질문을 검색용 문장 및 키워드로 변경
            vector_query, keyword_list = utill.rewrite_talk_question(USER_ID,user_query)
        print("="*60)
        print(vector_query)
        print(keyword_list)
        if vector_query and keyword_list:
            try:
                # 2. Qdrant 벡터 검색 (1단계: 관련 문서 후보 5개 추출)
                print("🔍 관련 자료를 검색하고 있습니다...")
                user_query_emb = utill.get_embedding(vector_query)
                #initial_docs = search_collection_data("test_cylee",user_query_emb, count=20)
                initial_docs = utill.search_collection_data_hybrid("test_cylee","full_contents",user_query_emb,keyword_list, 5)
                if not initial_docs:
                    print("⚠️ 검색된 기본 자료가 없습니다.")
                    continue
                """
                refined_docs = [point.payload.get("text", "") for point in initial_docs.points]
                # 3. 리랭커 필터링 (2단계: 점수 0.5 이상, 상위 3개 정밀 선별)
                # 이제 이 함수가 텍스트까지 포함된 리스트를 반환합니다.
                print("🎯 자료의 정확도를 분석 중입니다...")
                refined_query =  f"{user_query}. {keyword_list}"
                print(refined_query)
                refined_data = get_refined_context(
                    query=refined_query, 
                    documents=refined_docs, 
                    top_n=5, 
                    min_score=0.01
                )

                # 4. 컨텍스트 구성
                if not refined_data:
                    # 검색은 됐으나 점수가 너무 낮아 신뢰할 수 없는 경우
                    context_text = "" 
                    print("💡 참고할 만한 충분한 점수의 자료를 찾지 못했습니다.")
                    continue
                else:
                    # 검색된 텍스트들을 하나로 합침
                    context_text = "\n".join([initial_docs.points[item['index']].payload['full_contents'] for item in refined_data])
                    print(f"✅ {len(refined_data)}개의 핵심 근거를 찾았습니다.")
                """
                #받은 문장들을 하나의 변수로 저장
                context_text = "\n".join(point.payload.get("full_contents", "") for point in initial_docs.points)
                if context_text:
                    # 5. Ollama 답변 생성 (최종 단계)
                    print("✍️ 답변을 생성하는 중입니다. 잠시만 기다려 주세요...\n")
                    answer = utill.ask_ollama(user_query, context_text)
                    utill.update_memory('test', user_query, answer)
                    # 만약 답변에 '찾을 수 없습니다'가 포함되어 있다면, 그냥 문장 전체를 교체
                    if "정보를 찾을 수 없습니다" in answer:
                        answer = "정보를 찾을 수 없습니다."
                    elif "모르겠습니다" in answer:
                        answer = "질문에 대해 모르겠습니다."
                        
                    # 6. 결과 출력
                    print("-" * 40)
                    print(f"📢 상담관 답변:\n\n{answer}")
                    print("-" * 40)
                    """
                    # (선택 사항) 디버깅용 점수 출력
                    if refined_data:
                        scores = [f"{item['score']:.2f}" for item in refined_data]
                        print(f"참고 자료 신뢰도: {', '.join(scores)}")
                    """
                else:
                    answer = '질문에 대한 답변을 찾지 못하였습니다.'
                    utill.update_memory('test', user_query, answer)
                    # 6. 결과 출력
                    print("-" * 40)
                    print(f"📢 상담관 답변:\n\n{answer}")
                    print("-" * 40)

            except Exception as e:
                traceback_message = traceback.format_exc()
                print(traceback_message)
                print(f"❌ 오류가 발생했습니다: {e}")
        else:
            answer = '질문에 대한 답변을 찾지 못하였습니다.'
            utill.update_memory('test', user_query, answer)
            # 6. 결과 출력
            print("-" * 40)
            print(f"📢 상담관 답변:\n\n{answer}")
            print("-" * 40)

if __name__ == "__main__":
    run_consulting_system()