import asyncio
import json
import re
from typing import Union
from collections import Counter, defaultdict

# 从其他模块导入的一些辅助函数
from ._utils import (
    logger,  # 用于记录日志
    clean_str,  # 用于清理字符串
    compute_mdhash_id,  # 计算唯一ID的函数
    decode_tokens_by_tiktoken,  # 使用tiktoken解码tokens
    encode_string_by_tiktoken,  # 使用tiktoken编码字符串
    is_float_regex,  # 用正则表达式判断是否为浮点数
    list_of_list_to_csv,  # 将列表转换为CSV格式
    pack_user_ass_to_openai_messages,  # 将用户输入和响应打包为OpenAI格式
    split_string_by_multi_markers,  # 根据多个标记符拆分字符串
    truncate_list_by_token_size,  # 根据token大小截断列表
)

# 从其他模块导入的类和数据结构
from .base import (
    BaseGraphStorage,  # 基础图存储类，用于存储实体和关系
    BaseKVStorage,  # 键值存储类，用于存储数据
    BaseVectorStorage,  # 向量存储类，用于存储向量化数据
    SingleCommunitySchema,  # 单个社区的schema定义
    CommunitySchema,  # 社区的schema定义
    TextChunkSchema,  # 文本块的schema定义
    QueryParam,  # 查询参数的定义
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS  # 导入图表字段分隔符和提示模板

# 将内容根据token数量进行分块
def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)  # 编码字符串为tokens
    results = []  # 存储分块结果
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)  # 根据最大token大小和重叠部分分块
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )  # 解码分块的token为字符串
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),  # 当前块的token数量
                "content": chunk_content.strip(),  # 当前块的内容，去掉首尾空格
                "chunk_order_index": index,  # 当前块的顺序索引
            }
        )
    return results  # 返回分块结果

def chunking_by_token_size_batch(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    # 批量编码整个内容为tokens
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    
    # 计算所有的分块起始位置，考虑重叠部分
    chunks_indices = [
        (start, start + max_token_size) 
        for start in range(0, len(tokens), max_token_size - overlap_token_size)
    ]

    # 批量处理，准备分块
    results = []
    all_chunk_tokens = [tokens[start:end] for start, end in chunks_indices]
    
    # 批量解码这些tokens
    all_chunk_content = [
        decode_tokens_by_tiktoken(chunk, model_name=tiktoken_model).strip()
        for chunk in all_chunk_tokens
    ]
    
    # 生成最终的结果
    for index, chunk_content in enumerate(all_chunk_content):
        results.append(
            {
                "tokens": len(all_chunk_tokens[index]),  # 当前块的token数量
                "content": chunk_content,  # 当前块的内容
                "chunk_order_index": index,  # 当前块的顺序索引
            }
        )

    return results



# 处理实体或关系的描述，生成简化摘要
async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["cheap_model_func"]  # 获取使用的模型函数
    llm_max_tokens = global_config["cheap_model_max_token_size"]  # 最大token大小
    tiktoken_model_name = global_config["tiktoken_model_name"]  # tiktoken模型名
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]  # 摘要的最大token数量

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)  # 编码描述为tokens
    if len(tokens) < summary_max_tokens:  # 如果token数量小于摘要最大值，则无需摘要
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]  # 获取摘要的提示模板
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )  # 使用模型解码前llm_max_tokens个token
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),  # 按分隔符拆分描述
    )
    use_prompt = prompt_template.format(**context_base)  # 根据模板和上下文生成提示
    logger.debug(f"Trigger summary: {entity_or_relation_name}")  # 记录日志
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)  # 调用模型生成摘要
    return summary  # 返回摘要

# 处理单个实体的提取
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    # 如果不是实体或者记录长度不足，直接返回None
    if record_attributes[0] != '"entity"' or len(record_attributes) < 4:
        return None
    # 清理并提取实体名称、类型、描述和来源ID
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():  # 如果实体名称为空，返回None
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    # 返回实体的字典
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


# 处理单个关系的提取
async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    # 如果不是关系或者记录长度不足，直接返回None
    if record_attributes[0] != '"relationship"' or len(record_attributes) < 5:
        return None
    # 提取关系的源节点、目标节点、描述和权重等信息
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    # 如果最后一个属性是浮点数，作为权重；否则，权重默认为1.0
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    # 返回关系的字典
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


# 合并多个实体并将结果上载到图存储中
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    # 检查图存储中是否已经存在该实体
    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        # 如果存在，合并已有的实体类型、来源ID和描述
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    # 计算实体的主要类型（出现次数最多的类型）
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # 合并描述和来源ID
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    # 生成摘要
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )

    # 将实体更新到图存储
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data  # 返回合并后的实体数据



# 合并多个关系并将结果上载到图存储中
async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []

    # 检查是否已经存在这条边
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])

    # 合并权重、描述和来源ID
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )

    # 确保源节点和目标节点存在于图中
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )

    # 生成摘要并更新边数据
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            source_id=source_id,
        ),
    )


import time
from collections import defaultdict
import asyncio
from typing import Union

# 负责处理实体提取的主要异步任务
async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())  # 将文本块转为有序列表

    entity_extract_prompt = PROMPTS["entity_extraction"]  # 获取实体提取的提示模板
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],  # 默认元组分隔符
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],  # 默认记录分隔符
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],  # 默认完成分隔符
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),  # 默认实体类型
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]  # 提取继续的提示
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]  # 判断是否继续提取的提示

    already_processed = 0  # 已处理的块数量
    already_entities = 0  # 已提取的实体数量
    already_relations = 0  # 已提取的关系数量

    # 处理单个文本块内容
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        # 计时 - LLM 提取实体
        start_time = time.time()
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)  # 调用LLM模型提取实体
        end_time = time.time()
        # print(f"LLM entity extraction time for chunk {chunk_key}: {end_time - start_time:.2f} seconds")

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)  # 将提示与结果打包
        for now_glean_index in range(entity_extract_max_gleaning):  # 最多提取指定次数
            start_time = time.time()
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            end_time = time.time()
            # print(f"LLM gleaning time {now_glean_index + 1} for chunk {chunk_key}: {end_time - start_time:.2f} seconds")

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:  # 达到最大提取次数
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":  # 如果模型不建议继续提取，退出循环
                break

        # 将提取的结果根据记录和完成分隔符拆分
        start_time = time.time()
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        end_time = time.time()
        # print(f"String splitting time for chunk {chunk_key}: {end_time - start_time:.2f} seconds")

        maybe_nodes = defaultdict(list)  # 临时存储可能的节点
        maybe_edges = defaultdict(list)  # 临时存储可能的边

        start_time = time.time()
        for record in records:
            record = re.search(r"\((.*)\)", record)  # 使用正则提取括号内的内容
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )  # 提取实体
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )  # 提取关系
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        end_time = time.time()
        # print(f"Entity/Relationship extraction time for chunk {chunk_key}: {end_time - start_time:.2f} seconds")

        already_processed += 1
        already_entities += len(maybe_nodes)  # 记录已提取的实体数量
        already_relations += len(maybe_edges)  # 记录已提取的关系数量
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # 异步地处理所有文本块
    start_time = time.time()
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    end_time = time.time()
    print(f"Total time for processing all chunks: {end_time - start_time:.2f} seconds")
    
    print()  # 清除进度条
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:  # 合并提取结果
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # 处理无向图时，节点顺序需要排序
            maybe_edges[tuple(sorted(k))].extend(v)

    # 合并实体数据并更新到图存储
    start_time = time.time()
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    end_time = time.time()
    print(f"Total time for merging and upserting nodes: {end_time - start_time:.2f} seconds")

    # 合并关系数据并更新到图存储
    start_time = time.time()
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    end_time = time.time()
    print(f"Total time for merging and upserting edges: {end_time - start_time:.2f} seconds")

    # 如果没有提取到任何实体，记录警告日志
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None

    # 如果实体向量数据库不为空，将实体数据上载到向量数据库
    if entity_vdb is not None:
        start_time = time.time()
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
        end_time = time.time()
        print(f"Total time for upserting data to vector DB: {end_time - start_time:.2f} seconds")

    return knwoledge_graph_inst  # 返回更新后的图存储实例


# 将单个社区的子社区信息打包处理
def _pack_single_community_by_sub_communities(
    community: SingleCommunitySchema,  # 输入单个社区的schema
    max_token_size: int,  # 最大允许的token大小
    already_reports: dict[str, CommunitySchema],  # 已存在的社区报告
) -> tuple[str, int]:  # 返回子社区的描述、token数量、节点集合、边集合
    # 获取社区的子社区，并根据子社区的出现频率进行排序
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    
    # 根据max_token_size截断子社区描述，避免超过token限制
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],  # 使用报告字符串计算token大小
        max_token_size=max_token_size,
    )
    
    # 定义子社区描述的字段
    sub_fields = ["id", "report", "rating", "importance"]
    
    # 将子社区的描述转换为CSV格式，包括子社区ID、报告、评分和重要性
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,  # 子社区的索引
                c["report_string"],  # 子社区的报告字符串
                c["report_json"].get("rating", -1),  # 获取子社区的评分
                c["occurrence"],  # 子社区的出现频率
            ]
            for i, c in enumerate(may_trun_all_sub_communities)  # 遍历截断后的子社区
        ]
    )
    
    # 初始化用于存储子社区节点和边的列表
    already_nodes = []
    already_edges = []
    
    # 提取子社区中的节点和边，分别存储
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])  # 存储子社区的节点
        already_edges.extend([tuple(e) for e in c["edges"]])  # 存储子社区的边
    
    # 返回子社区的描述，token数量，包含的节点和边集合
    return (
        sub_communities_describe,  # 子社区描述
        len(encode_string_by_tiktoken(sub_communities_describe)),  # 描述的token数量
        set(already_nodes),  # 节点集合
        set(already_edges),  # 边集合
    )

# 打包单个社区的描述信息
async def _pack_single_community_describe(
    knwoledge_graph_inst: BaseGraphStorage,  # 知识图谱实例
    community: SingleCommunitySchema,  # 单个社区的schema
    max_token_size: int = 12000,  # 最大token大小
    already_reports: dict[str, CommunitySchema] = {},  # 已存在的社区报告
    global_config: dict = {},  # 全局配置
) -> str:
    # 将社区中的节点和边排序
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    # 异步获取每个节点和边的数据
    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    
    # 定义用于描述节点和边的字段
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    
    # 创建包含每个节点详细信息的列表（节点ID、实体、类型、描述和度数）
    nodes_list_data = [
        [
            i,  # 节点的索引
            node_name,  # 节点名称
            node_data.get("entity_type", "UNKNOWN"),  # 节点类型
            node_data.get("description", "UNKNOWN"),  # 节点描述
            await knwoledge_graph_inst.node_degree(node_name),  # 节点的度数
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    
    # 按照节点的度数进行降序排序
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    
    # 根据token大小对节点列表进行截断，确保总token数不超过限制的一半
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    
    # 创建包含每条边详细信息的列表（边ID、源节点、目标节点、描述和排名）
    edges_list_data = [
        [
            i,  # 边的索引
            edge_name[0],  # 边的源节点
            edge_name[1],  # 边的目标节点
            edge_data.get("description", "UNKNOWN"),  # 边的描述
            await knwoledge_graph_inst.edge_degree(*edge_name),  # 边的度数
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    
    # 按照边的度数进行降序排序
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    
    # 根据token大小对边列表进行截断，确保总token数不超过限制的一半
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    # 判断是否需要截断（如果节点或边列表被截断）
    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # 如果超出了token限制且社区有子社区，或强制使用子社区
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    
    # 如果需要或强制使用子社区，则打包子社区信息
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        
        # 排除不在子社区中的节点和边
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        
        # 如果报告的大小超过了max_token_size，节点和边将被设置为空
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    
    # 将截断后的节点和边信息转换为CSV格式
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    
    # 返回社区的报告描述、节点和边的描述
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


# 将社区报告的JSON转换为字符串格式
def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")  # 获取报告的标题
    summary = parsed_output.get("summary", "")  # 获取报告的摘要
    findings = parsed_output.get("findings", [])  # 获取报告的发现内容

    # 定义用于获取发现摘要和解释的函数
    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    # 将每个发现的摘要和解释拼接为报告的章节
    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    
    # 返回报告的标题、摘要和发现章节
    return f"# {title}\n\n{summary}\n\n{report_sections}"



# 生成社区报告
async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],  # 社区报告存储
    knwoledge_graph_inst: BaseGraphStorage,  # 知识图谱实例
    global_config: dict,  # 全局配置
):
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]  # LLM的额外参数
    use_llm_func: callable = global_config["best_model_func"]  # 使用的模型函数
    use_string_json_convert_func: callable = global_config[
        "convert_response_to_json_func"
    ]  # 将响应转换为JSON的函数

    community_report_prompt = PROMPTS["community_report"]  # 社区报告的提示模板

    # 获取社区的schema
    communities_schema = await knwoledge_graph_inst.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0  # 已处理的社区数

    # 生成单个社区的报告
    async def _form_single_community_report(
        community: SingleCommunitySchema, already_reports: dict[str, CommunitySchema]
    ):
        nonlocal already_processed
        # 打包单个社区的描述
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=already_reports,
            global_config=global_config,
        )
        # 根据描述生成社区报告
        # prompt = community_report_prompt.format(input_text=describe)
        # response = await use_llm_func(prompt, **llm_extra_kwargs)
        prompt = community_report_prompt.format(input_text=describe)
        # response = await use_llm_func(prompt, **llm_extra_kwargs)
        # 调用 LLM 函数，并等待返回结果
        response = await use_llm_func(prompt, **llm_extra_kwargs)
        
    #     last_brace_index = response.rfind('}')
    
    
    # # 截断字符串到最后一个 '}'
    #     response = response[:last_brace_index + 1]
    #     # 手动检查并修复缺少的 ] 和 }
    #     if not response.strip().endswith(']}'):
    #         # 如果 response 没有以 ]} 结尾，手动添加
    #         response = response.rstrip()  # 去除末尾的空白字符
    #         if not response.endswith(']'):
    #             response += ']'
    #         if not response.endswith(']}'):
    #             response += '}'
                
                
        # 将响应转换为数据
        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    # 根据社区的等级排序并生成报告
    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        # 更新社区报告数据
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    print()  # 清除进度条
    # 将社区报告上传到存储
    await community_report_kv.upsert(community_datas)


# 根据实体数据查找最相关的社区
async def _find_most_related_community_from_entities(
    node_datas: list[dict],  # 节点数据列表
    query_param: QueryParam,  # 查询参数，包含级别、最大token等限制
    community_reports: BaseKVStorage[CommunitySchema],  # 社区报告的存储
):
    related_communities = []  # 存储所有相关的社区
    for node_d in node_datas:  # 遍历节点数据
        if "clusters" not in node_d:  # 如果节点没有关联的社区，跳过
            continue
        related_communities.extend(json.loads(node_d["clusters"]))  # 提取相关的社区并扩展到列表

    # 过滤出符合查询参数级别的社区
    related_community_dup_keys = [
        str(dp["cluster"])  # 获取社区的唯一标识
        for dp in related_communities
        if dp["level"] <= query_param.level  # 只保留查询级别以下的社区
    ]
    
    # 统计每个社区的出现次数
    related_community_keys_counts = dict(Counter(related_community_dup_keys))
    
    # 异步获取所有相关社区的详细数据
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )
    
    # 过滤掉没有获取到数据的社区
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }
    
    # 根据社区的出现次数和评分对社区进行排序
    related_community_keys = sorted(
        related_community_keys_counts.keys(),
        key=lambda k: (
            related_community_keys_counts[k],  # 社区的出现次数
            related_community_datas[k]["report_json"].get("rating", -1),  # 社区的评分
        ),
        reverse=True,  # 降序排列
    )
    
    # 根据排序后的社区键获取社区数据
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    # 截断社区报告，确保不会超过最大token限制
    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],  # 使用报告字符串计算token大小
        max_token_size=query_param.local_max_token_for_community_report,  # 最大允许的token大小
    )
    
    # 如果只需要一个社区报告
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]
    
    return use_community_reports  # 返回找到的相关社区报告


# 查找与实体相关的文本单元
async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],  # 节点数据列表
    query_param: QueryParam,  # 查询参数，包含最大token等限制
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本块存储
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱存储
):
    # 将每个节点的来源ID进行拆分，获取所有文本单元
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    
    # 异步获取每个节点的边信息
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    
    all_one_hop_nodes = set()  # 存储所有一跳的节点
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])  # 添加一跳的目标节点到集合中
    
    all_one_hop_nodes = list(all_one_hop_nodes)
    
    # 异步获取一跳节点的数据
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    
    # 为每个一跳节点建立文本单元的映射表
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }

    all_text_units_lookup = {}  # 存储文本单元及其关联度信息
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                # 计算文本单元与一跳节点的关联次数
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            # 存储文本单元的数据、顺序和关联计数
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    
    # 检查是否有丢失的文本块数据
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    
    # 创建包含有效文本单元的数据列表
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    
    # 根据顺序和关联计数对文本单元进行排序
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    
    # 截断文本单元，确保不会超过最大token限制
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],  # 使用内容计算token大小
        max_token_size=query_param.local_max_token_for_text_unit,  # 最大允许的token大小
    )
    
    # 返回文本单元的列表
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


# 查找与实体相关的边
async def _find_most_related_edges_from_entities(
    node_datas: list[dict],  # 节点数据列表
    query_param: QueryParam,  # 查询参数，包含最大token等限制
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱存储
):
    # 异步获取所有节点的边
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    
    # 将边存储为集合，确保唯一性
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])  # 将边的节点排序，确保无向图的顺序一致
    
    all_edges = list(all_edges)  # 转换为列表以便后续处理
    
    # 异步获取所有边的数据
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    
    # 异步获取所有边的度数
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    
    # 将边的数据与度数结合
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    
    # 根据度数和权重对边进行排序
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    
    # 截断边的数据，确保不会超过最大token限制
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],  # 使用描述计算token大小
        max_token_size=query_param.local_max_token_for_local_context,  # 最大允许的token大小
    )
    
    # 返回排序后的边数据
    return all_edges_data


# 构建本地查询上下文
async def _build_local_query_context(
    query,  # 查询内容
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱存储
    entities_vdb: BaseVectorStorage,  # 实体向量数据库
    community_reports: BaseKVStorage[CommunitySchema],  # 社区报告存储
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本块存储
    query_param: QueryParam,  # 查询参数
):
    # 查询实体向量数据库，获取最相关的实体
    print("Querying entities...")
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    print(f"Found {len(results)} entities")
    if not len(results):  # 如果没有找到相关实体，返回None
        return None
    
    # 异步获取实体的节点数据
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    
    # 如果有缺失的节点数据，记录警告
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    
    # 异步获取节点的度数
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    
    # 将实体数据、节点数据和度数结合
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    
    # 查找与这些实体相关的社区、文本单元和关系
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    
    # 记录日志，显示使用了多少实体、社区、关系和文本单元
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    
    # 构建实体的描述部分
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,  # 实体的索引
                n["entity_name"],  # 实体名称
                n.get("entity_type", "UNKNOWN"),  # 实体类型
                n.get("description", "UNKNOWN"),  # 实体描述
                n["rank"],  # 实体的排名或度数
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)  # 将实体列表转换为CSV格式

    # 构建关系的描述部分
    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,  # 关系的索引
                e["src_tgt"][0],  # 源节点
                e["src_tgt"][1],  # 目标节点
                e["description"],  # 关系描述
                e["weight"],  # 关系权重
                e["rank"],  # 关系排名
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)  # 将关系列表转换为CSV格式

    # 构建社区的描述部分
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])  # 社区的报告内容
    communities_context = list_of_list_to_csv(communities_section_list)  # 将社区列表转换为CSV格式

    # 构建文本单元的描述部分
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])  # 文本单元内容
    text_units_context = list_of_list_to_csv(text_units_section_list)  # 将文本单元列表转换为CSV格式

    # 返回整个查询上下文，包含实体、关系、社区和文本单元的描述
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


# 本地查询的主要函数
async def local_query(
    query,  # 用户输入的查询
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱存储实例
    entities_vdb: BaseVectorStorage,  # 实体向量数据库
    community_reports: BaseKVStorage[CommunitySchema],  # 社区报告的键值存储
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本片段的键值存储
    query_param: QueryParam,  # 查询参数，包括最大token大小、是否只需要上下文等配置
    global_config: dict,  # 全局配置，包含模型调用和其他设置
) -> str:
    # 从全局配置中获取用于处理模型的函数
    use_model_func = global_config["best_model_func"]

    # 构建本地查询上下文，包含实体、文本块和社区等信息
    context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
    )
    
    # 如果只需要返回上下文而不需要进一步生成响应，则直接返回上下文
    if query_param.only_need_context:
        return context

    # 如果上下文为空，则返回失败提示
    if context is None:
        return PROMPTS["fail_response"]
    
    # 使用本地RAG（Retrieval-Augmented Generation）提示模板生成系统提示
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )

    # 调用模型函数生成最终响应
    response = await use_model_func(
        query,  # 用户查询
        system_prompt=sys_prompt,  # 系统提示（包括上下文）
    )
    return response  # 返回最终的模型响应


# 将社区数据分组，生成全局搜索的上下文映射
async def _map_global_communities(
    query: str,  # 用户的查询
    communities_data: list[CommunitySchema],  # 社区数据列表
    query_param: QueryParam,  # 查询参数
    global_config: dict,  # 全局配置
):
    # 使用全局配置中的函数将响应转换为JSON
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    # 使用的模型函数
    use_model_func = global_config["best_model_func"]
    community_groups = []

    # 分组社区数据，确保每组的总token数量不会超过限制
    while len(communities_data):
        # 根据报告字符串对社区进行截断，防止超过最大token限制
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
        )
        community_groups.append(this_group)
        # 更新剩余的社区数据
        communities_data = communities_data[len(this_group):]

    # 处理每个分组社区数据，生成全局社区映射的上下文
    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        # 构建社区的CSV描述，包括ID、内容、评分、重要性等字段
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,  # 社区的索引
                    c["report_string"],  # 社区的报告字符串
                    c["report_json"].get("rating", 0),  # 获取社区的评分
                    c["occurrence"],  # 社区的出现频率
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)  # 转换为CSV格式
        
        # 使用全局提示模板生成系统提示
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)

        
        # 调用模型函数处理当前组的社区映射
        response = await use_model_func(
            query,  # 用户查询
            system_prompt=sys_prompt,  # 生成的系统提示
            **query_param.global_special_community_map_llm_kwargs,  # 附加的特殊参数
        )
        data = use_string_json_convert_func(response)  # 将响应转换为JSON格式
        return data.get("points", [])  # 返回映射点的结果

    # 打印分组数量日志
    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    
    # 异步处理每个社区分组，收集所有结果
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    return responses  # 返回所有的处理结果


async def _map_global_communities_1(
    query: str,  # 用户的查询
    communities_data: list[CommunitySchema],  # 社区数据列表
    query_param: QueryParam,  # 查询参数
    global_config: dict,  # 全局配置
):
    # 使用全局配置中的函数将响应转换为JSON
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    # 使用的模型函数
    use_model_func = global_config["best_model_func"]
    community_groups = []

    # 分组社区数据，确保每组的总token数量不会超过限制
    while len(communities_data):
        # 根据报告字符串对社区进行截断，防止超过最大token限制
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_token_for_community_report,
        )
        community_groups.append(this_group)
        # 更新剩余的社区数据
        communities_data = communities_data[len(this_group):]

    # 处理每个分组社区数据，生成全局社区映射的上下文
    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        # 构建社区的CSV描述，包括ID、内容、评分、重要性等字段
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,  # 社区的索引
                    c["report_string"],  # 社区的报告字符串
                    c["report_json"].get("rating", 0),  # 获取社区的评分
                    c["occurrence"],  # 社区的出现频率
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)  # 转换为CSV格式
        
        # 使用全局提示模板生成系统提示
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)

        
        # 调用模型函数处理当前组的社区映射
        response = await use_model_func(
            query,  # 用户查询
            system_prompt=sys_prompt,  # 生成的系统提示
            **query_param.global_special_community_map_llm_kwargs,  # 附加的特殊参数
        )
        return response  # 返回映射点的结果

    # 打印分组数量日志
    logger.info(f"Grouping to {len(community_groups)} groups for global search")
    
    # 异步处理每个社区分组，收集所有结果
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    combined_response = ", ".join(responses)
    return combined_response  # 返回所有的处理结果



# 全局查询的主要函数
async def global_query(
    query,  # 用户输入的查询
    knowledge_graph_inst: BaseGraphStorage,  # 知识图谱存储实例
    entities_vdb: BaseVectorStorage,  # 实体向量数据库
    community_reports: BaseKVStorage[CommunitySchema],  # 社区报告的键值存储
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本片段的键值存储
    query_param: QueryParam,  # 查询参数
    global_config: dict,  # 全局配置
) -> str:
    # 获取社区的schema，过滤出级别符合的社区
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= query_param.level
    }
    # 如果没有符合条件的社区，返回失败提示
    if not len(community_schema):
        return PROMPTS["fail_response"]
    
    # 获取用于处理模型的函数
    use_model_func = global_config["best_model_func"]

    # 按照社区的出现频率排序
    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    # 取出配置中的最大社区数量
    sorted_community_schemas = sorted_community_schemas[
        : query_param.global_max_consider_community
    ]
    
    # 异步获取这些社区的报告数据
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    
    # 过滤掉没有报告的社区
    community_datas = [c for c in community_datas if c is not None]
    
    # 过滤掉评分低于全局最小评分要求的社区
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    
    # 按出现频率和评分进行排序
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    
    # 打印检索到的社区数量日志
    logger.info(f"Retrieved {len(community_datas)} communities")
    ####llama 3.1 8b指令follow能力太差，无法生成满足条件的json，直接把map的所有内容交给模型处理
    # map_communities_points = await _map_global_communities_1(
    #     query, community_datas, query_param, global_config
    # )
    # points_context = map_communities_points
    
    # 使用之前的函数映射全局社区
    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config
    )
    
    final_support_points = []
    # 整理每个分析员的支持点信息
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,  # 分析员的索引
                    "answer": point["description"],  # 支持点的描述
                    "score": point.get("score", 1),  # 支持点的评分
                }
            )
    
    # 过滤出评分大于0的支持点
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    
    # 如果没有支持点，返回失败提示
    if not len(final_support_points):
        return PROMPTS["fail_response"]
    
    # 根据评分对支持点进行降序排序
    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    
    # 截断支持点，确保不会超过最大token限制
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],  # 使用答案文本计算token大小
        max_token_size=query_param.global_max_token_for_community_report,  # 最大允许的token大小
    )
    
    # 将支持点内容转换为文本格式
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)  # 将支持点列表转换为字符串
    
    # 如果只需要上下文，则返回生成的支持点上下文
    if query_param.only_need_context:
        return points_context
    # 使用全局RAG提示模板生成系统提示
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    
    # 调用模型生成最终的响应
    response = await use_model_func(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=query_param.response_type
        ),
    )
    return response  # 返回最终的模型响应


# 简单查询函数，用于在没有复杂处理的情况下从文本片段中查询相关信息
async def naive_query(
    query,  # 用户查询
    chunks_vdb: BaseVectorStorage,  # 用于存储文本片段的向量数据库
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本片段的键值存储
    query_param: QueryParam,  # 查询参数，包含最大token大小、是否只需要上下文等设置
    global_config: dict,  # 全局配置，包含模型调用和其他设置信息
):
    # 从全局配置中获取用于调用LLM模型的函数
    use_model_func = global_config["best_model_func"]

    # 在文本片段向量数据库中根据用户查询获取最相关的片段，数量由top_k控制
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    
    # 如果没有找到任何相关片段，则返回失败响应
    if not len(results):
        return PROMPTS["fail_response"]
    
    # 提取查询结果中的文本片段ID
    chunks_ids = [r["id"] for r in results]

    # 根据这些ID从文本片段数据库中获取完整的文本片段
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # 根据最大token大小截断文本片段，以防超出限制
    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,  # 文本片段列表
        key=lambda x: x["content"],  # 使用文本片段的内容来计算token大小
        max_token_size=query_param.naive_max_token_for_text_unit,  # 最大允许的token数量
    )
    
    # 记录日志信息，显示截断前后的文本片段数量
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")

    # 将截断后的文本片段的内容拼接成一个完整的字符串，用于上下文或模型处理
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    # 如果查询参数指定只需要上下文内容而不需要调用LLM模型，则直接返回拼接的文本内容
    if query_param.only_need_context:
        return section

    # 使用"naive_rag_response"模板生成系统提示，内容为截断后的文本片段
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section,  # 上下文内容
        response_type=query_param.response_type  # 响应类型，可能用于控制输出格式
    )

    # 调用模型函数生成最终的响应
    response = await use_model_func(
        query,  # 用户查询
        system_prompt=sys_prompt,  # 模型的系统提示
    )
    
    # 返回模型生成的响应
    return response
