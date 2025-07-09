import torch
import clip
import numpy as np
from PIL import Image
import logging
from typing import List, Union, Tuple
import os
from tqdm import tqdm
import cv2

from config import CLIP_MODEL, CLIP_DEVICE, IMAGE_SIZE, NORMALIZE_EMBEDDINGS, SUPPORTED_IMAGE_EXTENSIONS

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPEncoder:
    """
    Classe para carregar e usar o modelo CLIP para extração de embeddings
    """
    
    def __init__(self, model_name: str = CLIP_MODEL, device: str = CLIP_DEVICE):
        """
        Inicializa o encoder CLIP
        
        Args:
            model_name: Nome do modelo CLIP
            device: Dispositivo (cpu ou cuda)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.preprocess = None
        
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo CLIP"""
        try:
            logger.info(f"Carregando modelo CLIP: {self.model_name}")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            logger.info(f"Modelo carregado com sucesso no dispositivo: {self.device}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo CLIP: {e}")
            raise
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Codifica uma imagem em embedding
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Embedding da imagem
        """
        try:
            # Carrega e preprocessa a imagem
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Gera embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                
                if NORMALIZE_EMBEDDINGS:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            logger.error(f"Erro ao codificar imagem {image_path}: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Codifica texto em embedding
        
        Args:
            text: Texto a ser codificado
            
        Returns:
            Embedding do texto
        """
        try:
            # Tokeniza o texto
            text_input = clip.tokenize([text]).to(self.device)
            
            # Gera embedding
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
                
                if NORMALIZE_EMBEDDINGS:
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().flatten()
        
        except Exception as e:
            logger.error(f"Erro ao codificar texto: {e}")
            raise
    
    def encode_images_batch(self, image_paths: List[str], 
                           batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """
        Codifica múltiplas imagens em lote
        
        Args:
            image_paths: Lista de caminhos das imagens
            batch_size: Tamanho do lote
            
        Returns:
            Embeddings das imagens e lista de caminhos válidos
        """
        embeddings = []
        valid_paths = []
        
        logger.info(f"Codificando {len(image_paths)} imagens...")
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Codificando imagens"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_valid_paths = []
            
            # Carrega imagens do lote
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(self.preprocess(image))
                    batch_valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Erro ao processar imagem {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Processa lote
            try:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                with torch.no_grad():
                    batch_features = self.model.encode_image(batch_tensor)
                    
                    if NORMALIZE_EMBEDDINGS:
                        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                
                embeddings.extend(batch_features.cpu().numpy())
                valid_paths.extend(batch_valid_paths)
                
            except Exception as e:
                logger.error(f"Erro ao processar lote: {e}")
                continue
        
        return np.array(embeddings), valid_paths
    
    def get_image_paths(self, directory: str) -> List[str]:
        """
        Obtém lista de caminhos de imagens válidas em um diretório
        
        Args:
            directory: Diretório das imagens
            
        Returns:
            Lista de caminhos de imagens
        """
        image_paths = []
        
        if not os.path.exists(directory):
            logger.warning(f"Diretório não existe: {directory}")
            return image_paths
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS):
                    image_paths.append(os.path.join(root, file))
        
        logger.info(f"Encontradas {len(image_paths)} imagens em {directory}")
        return image_paths
    
    def calculate_similarity(self, embedding1: np.ndarray, 
                           embedding2: np.ndarray) -> float:
        """
        Calcula similaridade cosseno entre dois embeddings
        
        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding
            
        Returns:
            Similaridade cosseno
        """
        # Normaliza embeddings se necessário
        if not NORMALIZE_EMBEDDINGS:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        return np.dot(embedding1, embedding2)
    
    def get_embedding_dimension(self) -> int:
        """
        Retorna a dimensão dos embeddings
        
        Returns:
            Dimensão dos embeddings
        """
        # Testa com um texto simples
        test_embedding = self.encode_text("test")
        return test_embedding.shape[0] 