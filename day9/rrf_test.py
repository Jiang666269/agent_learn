
#reciprocal rank fusion
# BM25 检索结果，排名从高到低
bm25_results = ["A","C","B","D"]
vector_results = ["D","B","C","A"]

#k：平滑度常数，一般60
def rrf_fusion(result_lists,k=60):
    scores = {}

    #遍历每一种检索方法的结果
    for results in result_lists:

        #排名从1开始
        #enumerate 同时获得文档排名和ID
        for rank,doc_id in enumerate(results,start=1):

            #第一次遇到文档时初始化得分
            if doc_id not in scores:
                scores[doc_id] = 0.0

            #累加RRF得分
            scores[doc_id] += 1 / (k+rank)

    #按综合得分从高到低排序
    ranked_results = sorted(
        scores.items(),
        key=lambda item:item[1],
        reverse = True
    )

    return ranked_results

results = rrf_fusion([bm25_results,vector_results])

for doc_id,score in results:
    print(f"文档{doc_id}:{score:.6f}")