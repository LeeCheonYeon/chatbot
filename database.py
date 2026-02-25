from utill import load_from_view, create_collection, update_collection_data, trans_list_to_pointStructList, create_index
#콜렉션 생성
create_collection('test_cylee')

#데이터 가져오기
result = load_from_view()

#데이터 타입 변경
trans_result = trans_list_to_pointStructList(result,'B')

#데이터 저장
update_collection_data('test_cylee',trans_result)

#인덱스 생성
create_index('test_cylee','full_contents')


