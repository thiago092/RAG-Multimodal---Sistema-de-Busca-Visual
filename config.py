import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configurações gerais
APP_TITLE = "RAG Multimodal - Trabalho de Estrutura de Dados"
APP_DESCRIPTION = "Sistema de recuperação de imagens baseado em texto com CLIP + HNSW + LLM Multimodal"

# Diretórios
IMAGES_DIR = "images"
INDEX_DIR = "index"
CACHE_DIR = "cache"
RESULTS_DIR = "results"

# Configurações do CLIP
CLIP_MODEL = "ViT-B/32"  # Modelo CLIP a ser usado
CLIP_DEVICE = "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"

# Configurações do HNSW
HNSW_SPACE = "cosine"  # Espaço de distância (cosine, l2, ip)
HNSW_EF_CONSTRUCTION = 200  # Parâmetro de construção
HNSW_EF_SEARCH = 50  # Parâmetro de busca
HNSW_M = 16  # Número de conexões bidirecionais
HNSW_MAX_ELEMENTS = 10000  # Número máximo de elementos

# Configurações do LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyARH9MHjSTl-CEBq30OuicNZG_ivBa_u1c")
GEMINI_MODEL = "gemini-1.5-flash"  # Modelo do Google Gemini
MAX_TOKENS = 1000
TEMPERATURE = 0.7

# Configurações de busca
TOP_K_RETRIEVAL = 3  # Número de imagens a recuperar
SIMILARITY_THRESHOLD = 0.2  # Threshold de similaridade (ajustado para garantir resultados)

# Configurações de métricas
METRICS_ENABLED = True
SAVE_RESULTS = True
SHOW_IMAGES = True

# Configurações da interface
STREAMLIT_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": "🖼️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Configurações de logs
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Extensões de imagem suportadas
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# Configurações de processamento
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NORMALIZE_EMBEDDINGS = True 