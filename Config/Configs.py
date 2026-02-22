import logging
import os
import faiss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

def ConfigValues(service="Search"):


    # Documents
    DocFolder = "./Documents"
    DocPath = f"{DocFolder}/{service}"
    PdfPath = f"{DocPath}.pdf"
    DocxPath = f"{DocPath}.docx"

    # Database
    DBFolder = "./Database"
    DBPath = f"{DBFolder}/{service}/{service}"

    RawExtractPath = f"{DBPath}_Extract"
    ChunksPath = f"{DBPath}_Chunks"
    EmbeddingPath = f"{DBPath}_Embedding"

    RawDataPath = f"{RawExtractPath}_Raw.json"
    RawLvlsPath = f"{RawExtractPath}_Levels.json"

    StructPath = f"{ChunksPath}_Struct.json"
    SegmentPath = f"{ChunksPath}_Segment.json"
    SchemaPath = f"{ChunksPath}_Schema.json"
    
    FaissPath = f"{EmbeddingPath}_Index.faiss"
    MappingPath = f"{EmbeddingPath}_Mapping.json"
    MapDataPath = f"{EmbeddingPath}_MapData.json"
    MapChunkPath = f"{EmbeddingPath}_MapChunk.json"
    MetaPath = f"{EmbeddingPath}_Meta.json"

    # Keys
    DATA_KEY = "contents"
    EMBE_KEY = "embeddings"

    # Models
    SEARCH_EGINE = faiss.IndexFlatIP
    RERANK_MODEL = "BAAI/bge-reranker-base"
    EMBEDD_MODEL = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    RESPON_MODEL = "gpt-3.5-turbo"

    WORD_LIMIT = 1000
    print("=== Configuration Loaded ===")
    print(f"Service : {service}")
    print(f"Document: {PdfPath}")
    print(f"Raw Data: {RawDataPath}")
    print(f"Raw Lvls: {RawLvlsPath}")
    print(f"Struct  : {StructPath}")
    print(f"Segment : {SegmentPath}")
    print(f"Schema  : {SchemaPath}")
    print(f"Faiss   : {FaissPath}")
    print(f"Mapping : {MappingPath}")
    print(f"MapData : {MapDataPath}")
    print(f"MapChunk: {MapChunkPath}")
    print(f"Meta    : {MetaPath}")
    print(f"Embedder: {EMBEDD_MODEL}")
    print(f"Searcher: {SEARCH_EGINE}")
    print(f"Reranker: {RERANK_MODEL}")
    print(f"Responer: {RESPON_MODEL}")
    print(f"Data Key: {DATA_KEY}")
    print(f"Embe Key: {EMBE_KEY}")
    print(f"Word Lim: {WORD_LIMIT}")
    print("============================")

    return {
        "PdfPath": PdfPath,
        "DocxPath": DocxPath,
        "RawDataPath": RawDataPath,
        "RawLvlsPath": RawLvlsPath,
        "StructPath": StructPath,
        "SegmentPath": SegmentPath,
        "SchemaPath": SchemaPath,
        "FaissPath": FaissPath,
        "MappingPath": MappingPath,
        "MapDataPath": MapDataPath,
        "MapChunkPath": MapChunkPath,
        "MetaPath": MetaPath,
        "DATA_KEY": DATA_KEY,
        "EMBE_KEY": EMBE_KEY,
        "EMBEDD_MODEL": EMBEDD_MODEL,
        "SEARCH_EGINE": SEARCH_EGINE,
        "RERANK_MODEL": RERANK_MODEL,
        "RESPON_MODEL": RESPON_MODEL,
        "WORD_LIMIT": WORD_LIMIT
    }
