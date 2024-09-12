import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

# 导入自定义模块中的相关功能
from ._llm import gpt_4o_complete, gpt_4o_mini_complete, openai_embedding
from ._op import (
    chunking_by_token_size,  # 文本分块功能
    extract_entities,  # 实体提取功能
    generate_community_report,  # 生成社区报告功能
    local_query,  # 本地查询
    global_query,  # 全局查询
    naive_query,  # 简单查询
)
from ._storage import (
    JsonKVStorage,  # JSON键值存储
    NanoVectorDBStorage,  # 矢量数据库存储
    NetworkXStorage,  # 网络图存储
)
from ._utils import (
    EmbeddingFunc,  # 嵌入函数
    compute_mdhash_id,  # 计算MD5哈希ID
    limit_async_func_call,  # 限制异步函数调用数量
    convert_response_to_json,  # 将响应转换为JSON格式
    logger,  # 日志记录器
)
from .base import (
    BaseGraphStorage,  # 基础图存储类
    BaseKVStorage,  # 基础键值存储类
    BaseVectorStorage,  # 基础矢量存储类
    StorageNameSpace,  # 存储命名空间
    QueryParam,  # 查询参数
)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    获取当前的异步事件循环。如果在子线程中无法获取事件循环，则创建一个新的事件循环。
    """
    try:
        # 如果已有事件循环，直接获取它
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 如果在子线程中没有事件循环，创建新的事件循环
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# 使用dataclass定义GraphRAG类，用于管理图结构的存储和相关操作
@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    enable_local: bool = True  # 是否启用本地查询
    enable_naive_rag: bool = False  # 是否启用简单RAG模式

    chunk_token_size: int = 1200  # 文本分块的token大小
    chunk_overlap_token_size: int = 100  # 文本分块重叠的token大小
    tiktoken_model_name: str = "gpt-4o"  # 使用的token化模型名称

    entity_extract_max_gleaning: int = 1  # 实体提取的最大迭代次数
    entity_summary_to_max_tokens: int = 500  # 实体摘要最大token数

    graph_cluster_algorithm: str = "leiden"  # 使用的图聚类算法
    max_graph_cluster_size: int = 10  # 图聚类的最大尺寸
    graph_cluster_seed: int = 0xDEADBEEF  # 图聚类的随机种子

    node_embedding_algorithm: str = "node2vec"  # 节点嵌入算法
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # 特殊社区报告的LLM参数
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)  # 嵌入函数
    embedding_batch_num: int = 32  # 嵌入批次大小
    embedding_func_max_async: int = 16  # 最大异步嵌入调用次数

    # 使用的模型配置
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768  # 模型的最大token数
    best_model_max_async: int = 16  # 最大异步调用数量
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # 存储系统配置
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage  # JSON键值存储类
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage  # 矢量数据库存储类
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage  # 图存储类
    enable_llm_cache: bool = True  # 是否启用LLM缓存

    # 扩展参数
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json  # 响应转换为JSON的函数

    def __post_init__(self):
        """
        初始化GraphRAG实例，并配置相关的工作目录、存储和功能。
        """
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        # 创建工作目录
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # 配置存储系统实例
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        # 如果启用了LLM缓存，配置LLM缓存
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        # 限制异步调用次数的嵌入函数
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        # 配置实体和分块的矢量数据库
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        # 限制异步调用次数的模型函数
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def insert(self, string_or_strings):
        """
        插入文档数据，并进行处理（异步方法的同步包装）。
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        """
        查询数据，并返回结果（异步方法的同步包装）。
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    def eval(self, querys: list[str], contexts: list[str], answers: list[str]):
        """
        执行评估任务（异步方法的同步包装）。
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aeval(querys, contexts, answers))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        异步查询方法，支持本地、全局和简单模式。
        """
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in local mode")
        
        # 根据模式执行不同类型的查询
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        """
        异步插入文档，处理文档的分块、实体提取、图更新和社区报告生成。
        """
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
            
            # 生成新文档的哈希ID
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            
            # 过滤已有文档，避免重复插入
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- 文本分块
            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            
            # 过滤已有分块，避免重复插入
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: 目前不支持社区的增量更新，因此清除所有已有报告
            await self.community_reports.drop()

            # ---------- 实体提取并更新到图存储
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            
            self.chunk_entity_relation_graph = maybe_new_kg
            
            # ---------- 图的聚类和社区报告生成
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )

            # ---------- 提交插入和索引更新
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def aeval(self, querys: list[str], contexts: list[str], answers: list[str]):
        """
        异步评估方法，尚未实现功能。
        """
        pass

    async def _insert_done(self):
        """
        插入完成后的回调，更新所有存储的索引。
        """
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        """
        查询完成后的回调，更新LLM缓存的索引。
        """
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
