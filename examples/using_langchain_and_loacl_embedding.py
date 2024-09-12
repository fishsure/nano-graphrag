import os
import logging
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
from sentence_transformers import SentenceTransformer
from nano_graphrag._utils import wrap_embedding_func_with_attrs
import time

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

MODEL = "Meta-Llama-3.1-8B-Instruct"
WORKING_DIR = "./nano_graphrag_langchain_TEST"

EMBED_MODEL = SentenceTransformer(
    "bge-m3", cache_folder=WORKING_DIR, device="cuda:1"
)
print("EMBED_MODEL loaded: ", EMBED_MODEL.get_sentence_embedding_dimension())

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

from langchain_openai import OpenAI, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

async def langchain_model(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    **kwargs
) -> str:
    llm = ChatOpenAI(
        model_name='Meta-Llama-3.1-8B-Instruct',  # 使用的模型名称
        api_key='your-api-key',  # 替换为实际的 API 密钥
        max_tokens=4000,
        temperature=0.0,
        top_p=1.0,
        base_url='http://localhost:8000/v1/',  # 替换为实际的模型服务地址
        max_retries=10,
    )

    prompt_template = PromptTemplate.from_template(
        "System Message: {system_message}\nUser: {user_prompt}\n"
    )
    
    system_message = system_prompt if system_prompt else "No system message"
    user_prompt = prompt  
    
    chain = prompt_template | llm
    response = chain.invoke({
        "system_message": system_message,
        "user_prompt": user_prompt,
    })

    result = response.content
    return result

def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

def query():
    start_time = time.time()  # 记录开始时间
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=langchain_model,
        cheap_model_func=langchain_model,
        embedding_func=local_embedding,
    )
    
    print("Running query...")
    query_start = time.time()  # 记录查询部分开始时间
    result = rag.query(
        "What are the top themes in this story?", param=QueryParam(mode="global")
    )
    query_end = time.time()  # 记录查询部分结束时间
    print(f"Query result: {result}")
    
    end_time = time.time()  # 记录结束时间
    print(f"Total query time: {end_time - start_time:.2f} seconds")
    print(f"Time spent on query itself: {query_end - query_start:.2f} seconds")

def insert():
    from time import time

    with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=langchain_model,
        cheap_model_func=langchain_model,
        embedding_func=local_embedding,
    )

    print("Starting insert...")
    start = time()
    rag.insert(FAKE_TEXT)
    end = time()
    print(f"Indexing time: {end - start:.2f} seconds")

if __name__ == "__main__":
    print("Running insert...")
    insert_time_start = time.time()  # 记录整个插入过程的开始时间
    insert()
    insert_time_end = time.time()  # 记录整个插入过程的结束时间
    print(f"Total insert time: {insert_time_end - insert_time_start:.2f} seconds")
    
    print("\nRunning query...")
    query_time_start = time.time()  # 记录整个查询过程的开始时间
    query()
    query_time_end = time.time()  # 记录整个查询过程的结束时间
    print(f"Total query execution time: {query_time_end - query_time_start:.2f} seconds")
