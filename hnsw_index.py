import hnswlib
import numpy as np
import pickle
import os
import logging
from typing import List, Tuple, Dict, Any
import json
from pathlib import Path

from config import (
    HNSW_SPACE, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, 
    HNSW_M, HNSW_MAX_ELEMENTS, INDEX_DIR, TOP_K_RETRIEVAL
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HNSWIndex:
    """
    Classe para gerenciar índice HNSW para busca vetorial
    """
    
    def __init__(self, dimension: int, space: str = HNSW_SPACE):
        """
        Inicializa o índice HNSW
        
        Args:
            dimension: Dimensão dos vetores
            space: Espaço de distância ('cosine', 'l2', 'ip')
        """
        self.dimension = dimension
        self.space = space
        self.index = None
        self.metadata = {}
        self.is_built = False
        
        # Criar diretório de índices se não existir
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        self._create_index()
    
    def _create_index(self):
        """Cria um novo índice HNSW"""
        try:
            logger.info(f"Criando índice HNSW com dimensão {self.dimension} e espaço {self.space}")
            self.index = hnswlib.Index(space=self.space, dim=self.dimension)
            self.index.init_index(
                max_elements=HNSW_MAX_ELEMENTS,
                ef_construction=HNSW_EF_CONSTRUCTION,
                M=HNSW_M
            )
            self.index.set_ef(HNSW_EF_SEARCH)
            logger.info("Índice HNSW criado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao criar índice HNSW: {e}")
            raise
    
    def add_items(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Adiciona itens ao índice
        
        Args:
            embeddings: Array de embeddings
            metadata: Lista de metadados correspondentes
        """
        try:
            if len(embeddings) != len(metadata):
                raise ValueError("Número de embeddings deve ser igual ao número de metadados")
            
            # Gera IDs únicos para os itens
            start_id = len(self.metadata)
            ids = list(range(start_id, start_id + len(embeddings)))
            
            # Adiciona embeddings ao índice
            logger.info(f"Adicionando {len(embeddings)} itens ao índice...")
            self.index.add_items(embeddings, ids)
            
            # Salva metadados
            for i, meta in enumerate(metadata):
                self.metadata[start_id + i] = meta
            
            self.is_built = True
            logger.info(f"Itens adicionados com sucesso. Total: {len(self.metadata)}")
            
        except Exception as e:
            logger.error(f"Erro ao adicionar itens: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = TOP_K_RETRIEVAL) -> Tuple[List[int], List[float]]:
        """
        Busca os k itens mais similares
        
        Args:
            query_embedding: Embedding da consulta
            k: Número de itens a retornar
            
        Returns:
            Lista de IDs e lista de distâncias
        """
        try:
            if not self.is_built:
                raise ValueError("Índice não foi construído ainda")
            
            # Realiza busca
            labels, distances = self.index.knn_query(query_embedding, k=k)
            
            # Converte para listas se necessário
            if isinstance(labels, np.ndarray):
                labels = labels.flatten().tolist()
            if isinstance(distances, np.ndarray):
                distances = distances.flatten().tolist()
            
            return labels, distances
            
        except Exception as e:
            logger.error(f"Erro na busca: {e}")
            raise
    
    def get_metadata(self, item_id: int) -> Dict[str, Any]:
        """
        Obtém metadados de um item
        
        Args:
            item_id: ID do item
            
        Returns:
            Metadados do item
        """
        return self.metadata.get(item_id, {})
    
    def get_search_results(self, query_embedding: np.ndarray, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Busca e retorna resultados com metadados
        
        Args:
            query_embedding: Embedding da consulta
            k: Número de itens a retornar
            
        Returns:
            Lista de resultados com metadados e similaridade
        """
        try:
            labels, distances = self.search(query_embedding, k)
            
            results = []
            for label, distance in zip(labels, distances):
                metadata = self.get_metadata(label)
                
                # Converte distância para similaridade (para cosine)
                if self.space == 'cosine':
                    similarity = 1.0 - distance
                elif self.space == 'l2':
                    similarity = 1.0 / (1.0 + distance)
                else:  # inner product
                    similarity = distance
                
                result = {
                    'id': label,
                    'distance': distance,
                    'similarity': similarity,
                    'metadata': metadata
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Erro ao obter resultados: {e}")
            raise
    
    def save_index(self, filename: str):
        """
        Salva o índice em arquivo
        
        Args:
            filename: Nome do arquivo (sem extensão)
        """
        try:
            index_path = os.path.join(INDEX_DIR, f"{filename}.bin")
            metadata_path = os.path.join(INDEX_DIR, f"{filename}_metadata.pkl")
            config_path = os.path.join(INDEX_DIR, f"{filename}_config.json")
            
            # Salva índice HNSW
            self.index.save_index(index_path)
            
            # Salva metadados
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            # Salva configuração
            config = {
                'dimension': self.dimension,
                'space': self.space,
                'num_elements': len(self.metadata),
                'ef_construction': HNSW_EF_CONSTRUCTION,
                'ef_search': HNSW_EF_SEARCH,
                'M': HNSW_M
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Índice salvo: {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar índice: {e}")
            raise
    
    def load_index(self, filename: str):
        """
        Carrega índice de arquivo
        
        Args:
            filename: Nome do arquivo (sem extensão)
        """
        try:
            index_path = os.path.join(INDEX_DIR, f"{filename}.bin")
            metadata_path = os.path.join(INDEX_DIR, f"{filename}_metadata.pkl")
            config_path = os.path.join(INDEX_DIR, f"{filename}_config.json")
            
            # Verifica se arquivos existem
            if not all(os.path.exists(p) for p in [index_path, metadata_path, config_path]):
                raise FileNotFoundError(f"Arquivos do índice {filename} não encontrados")
            
            # Carrega configuração
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Recria índice com configuração salva
            self.dimension = config['dimension']
            self.space = config['space']
            self._create_index()
            
            # Carrega índice HNSW
            self.index.load_index(index_path, max_elements=config['num_elements'])
            
            # Carrega metadados
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.is_built = True
            logger.info(f"Índice carregado: {filename} ({len(self.metadata)} itens)")
            
        except Exception as e:
            logger.error(f"Erro ao carregar índice: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do índice
        
        Returns:
            Dicionário com estatísticas
        """
        stats = {
            'num_elements': len(self.metadata),
            'dimension': self.dimension,
            'space': self.space,
            'is_built': self.is_built,
            'ef_construction': HNSW_EF_CONSTRUCTION,
            'ef_search': HNSW_EF_SEARCH,
            'M': HNSW_M,
            'max_elements': HNSW_MAX_ELEMENTS
        }
        return stats
    
    def list_saved_indexes(self) -> List[str]:
        """
        Lista índices salvos
        
        Returns:
            Lista de nomes de índices salvos
        """
        if not os.path.exists(INDEX_DIR):
            return []
        
        indexes = []
        for filename in os.listdir(INDEX_DIR):
            if filename.endswith('.bin'):
                name = filename[:-4]  # Remove .bin
                indexes.append(name)
        
        return indexes
    
    def delete_index(self, filename: str):
        """
        Deleta um índice salvo
        
        Args:
            filename: Nome do arquivo (sem extensão)
        """
        try:
            index_path = os.path.join(INDEX_DIR, f"{filename}.bin")
            metadata_path = os.path.join(INDEX_DIR, f"{filename}_metadata.pkl")
            config_path = os.path.join(INDEX_DIR, f"{filename}_config.json")
            
            for path in [index_path, metadata_path, config_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            logger.info(f"Índice deletado: {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao deletar índice: {e}")
            raise 