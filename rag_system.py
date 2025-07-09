import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from clip_encoder import CLIPEncoder
from hnsw_index import HNSWIndex
from llm_multimodal import MultimodalLLM
from metrics import RAGMetrics
from config import (
    IMAGES_DIR, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD, 
    BATCH_SIZE, METRICS_ENABLED
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGMultimodalSystem:
    """
    Sistema RAG Multimodal completo integrando CLIP, HNSW e LLM
    """
    
    def __init__(self, images_directory: str = IMAGES_DIR):
        """
        Inicializa o sistema RAG
        
        Args:
            images_directory: Diretório com as imagens
        """
        self.images_directory = images_directory
        self.clip_encoder = None
        self.hnsw_index = None
        self.llm = None
        self.metrics = RAGMetrics() if METRICS_ENABLED else None
        self.is_initialized = False
        
        # Cria diretório de imagens se não existir
        os.makedirs(images_directory, exist_ok=True)
        
        logger.info("Sistema RAG Multimodal inicializado")
    
    def initialize_components(self):
        """Inicializa todos os componentes do sistema"""
        try:
            logger.info("Inicializando componentes do sistema...")
            
            # Inicializa CLIP
            logger.info("Carregando modelo CLIP...")
            self.clip_encoder = CLIPEncoder()
            
            # Inicializa LLM
            logger.info("Configurando LLM multimodal...")
            self.llm = MultimodalLLM()
            
            # Obtém dimensão dos embeddings
            embedding_dim = self.clip_encoder.get_embedding_dimension()
            logger.info(f"Dimensão dos embeddings: {embedding_dim}")
            
            # Inicializa índice HNSW
            logger.info("Criando índice HNSW...")
            self.hnsw_index = HNSWIndex(dimension=embedding_dim)
            
            self.is_initialized = True
            logger.info("Todos os componentes inicializados com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar componentes: {e}")
            raise
    
    def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Constrói o índice HNSW com as imagens
        
        Args:
            force_rebuild: Force reconstrução do índice
            
        Returns:
            Estatísticas da construção do índice
        """
        if not self.is_initialized:
            self.initialize_components()
        
        try:
            logger.info("Iniciando construção do índice...")
            start_time = time.time()
            
            # Obtém lista de imagens
            image_paths = self.clip_encoder.get_image_paths(self.images_directory)
            
            if not image_paths:
                raise ValueError(f"Nenhuma imagem encontrada em {self.images_directory}")
            
            logger.info(f"Encontradas {len(image_paths)} imagens")
            
            # Extrai embeddings das imagens
            logger.info("Extraindo embeddings das imagens...")
            embedding_start = time.time()
            
            embeddings, valid_paths = self.clip_encoder.encode_images_batch(
                image_paths, 
                batch_size=BATCH_SIZE
            )
            
            embedding_time = time.time() - embedding_start
            
            if len(embeddings) == 0:
                raise ValueError("Nenhum embedding foi extraído das imagens")
            
            # Cria metadados para cada imagem
            metadata = []
            for i, path in enumerate(valid_paths):
                meta = {
                    'image_path': path,
                    'filename': os.path.basename(path),
                    'directory': os.path.dirname(path),
                    'file_size': os.path.getsize(path),
                    'embedding_index': i
                }
                metadata.append(meta)
            
            # Adiciona ao índice HNSW
            logger.info("Adicionando embeddings ao índice HNSW...")
            index_start = time.time()
            
            self.hnsw_index.add_items(embeddings, metadata)
            
            index_time = time.time() - index_start
            total_time = time.time() - start_time
            
            # Estatísticas
            stats = {
                'total_images': len(image_paths),
                'successful_embeddings': len(embeddings),
                'failed_embeddings': len(image_paths) - len(embeddings),
                'embedding_time': embedding_time,
                'indexing_time': index_time,
                'total_time': total_time,
                'embeddings_per_second': len(embeddings) / embedding_time if embedding_time > 0 else 0,
                'index_statistics': self.hnsw_index.get_statistics()
            }
            
            logger.info(f"Índice construído com sucesso: {len(embeddings)} imagens em {total_time:.2f}s")
            return stats
            
        except Exception as e:
            logger.error(f"Erro ao construir índice: {e}")
            raise
    
    def search_images(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Busca imagens similares baseadas em consulta textual
        
        Args:
            query: Consulta textual
            k: Número de resultados a retornar
            
        Returns:
            Lista de resultados ordenados por similaridade
        """
        if not self.is_initialized:
            raise ValueError("Sistema não foi inicializado. Chame initialize_components() primeiro.")
        
        if not self.hnsw_index.is_built:
            raise ValueError("Índice não foi construído. Chame build_index() primeiro.")
        
        try:
            logger.info(f"Buscando imagens para consulta: '{query}'")
            
            # Codifica consulta textual
            query_embedding = self.clip_encoder.encode_text(query)
            
            # Busca no índice HNSW
            results = self.hnsw_index.get_search_results(query_embedding, k)
            
            # Filtra por threshold de similaridade
            filtered_results = [
                r for r in results 
                if r['similarity'] >= SIMILARITY_THRESHOLD
            ]
            
            logger.info(f"Encontrados {len(filtered_results)} resultados acima do threshold")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Erro na busca: {e}")
            raise
    
    def generate_responses(self, query: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Gera respostas usando LLM multimodal
        
        Args:
            query: Consulta original
            search_results: Resultados da busca
            
        Returns:
            Lista de respostas geradas
        """
        if not self.llm:
            raise ValueError("LLM não foi inicializado")
        
        try:
            logger.info(f"Gerando respostas para {len(search_results)} resultados")
            
            responses = self.llm.generate_batch_responses(search_results, query)
            
            return responses
            
        except Exception as e:
            logger.error(f"Erro ao gerar respostas: {e}")
            raise
    
    def query(self, query: str, k: int = TOP_K_RETRIEVAL, 
              generate_responses: bool = True) -> Dict[str, Any]:
        """
        Executa consulta completa no sistema RAG
        
        Args:
            query: Consulta textual
            k: Número de resultados a recuperar
            generate_responses: Se deve gerar respostas com LLM
            
        Returns:
            Resultado completo da consulta
        """
        if not self.is_initialized:
            raise ValueError("Sistema não foi inicializado")
        
        # Inicia timer para métricas
        start_time = time.time() if self.metrics else None
        
        try:
            logger.info(f"Executando consulta: '{query}'")
            
            # Busca imagens similares
            search_start = time.time()
            search_results = self.search_images(query, k)
            search_time = time.time() - search_start
            
            # Gera respostas se solicitado
            responses = []
            llm_time = 0
            
            if generate_responses and search_results:
                llm_start = time.time()
                responses = self.generate_responses(query, search_results)
                llm_time = time.time() - llm_start
            
            # Resultado final
            result = {
                'query': query,
                'search_results': search_results,
                'responses': responses,
                'num_results': len(search_results),
                'num_responses': len(responses),
                'timing': {
                    'search_time': search_time,
                    'llm_time': llm_time,
                    'total_time': time.time() - start_time if start_time else 0
                },
                'success': True
            }
            
            # Registra métricas
            if self.metrics and start_time:
                metadata = {
                    'hnsw_search_time': search_time,
                    'llm_generation_time': llm_time
                }
                self.metrics.record_query_metrics(
                    query, search_results, responses, start_time, metadata
                )
            
            logger.info(f"Consulta executada com sucesso: {len(search_results)} resultados")
            return result
            
        except Exception as e:
            logger.error(f"Erro na consulta: {e}")
            return {
                'query': query,
                'search_results': [],
                'responses': [],
                'num_results': 0,
                'num_responses': 0,
                'timing': {'total_time': time.time() - start_time if start_time else 0},
                'success': False,
                'error': str(e)
            }
    
    def save_index(self, filename: str = "multimodal_rag_index"):
        """
        Salva o índice atual
        
        Args:
            filename: Nome do arquivo para salvar
        """
        if not self.hnsw_index or not self.hnsw_index.is_built:
            raise ValueError("Índice não foi construído ainda")
        
        try:
            self.hnsw_index.save_index(filename)
            logger.info(f"Índice salvo: {filename}")
        except Exception as e:
            logger.error(f"Erro ao salvar índice: {e}")
            raise
    
    def load_index(self, filename: str = "multimodal_rag_index"):
        """
        Carrega índice salvo
        
        Args:
            filename: Nome do arquivo para carregar
        """
        if not self.is_initialized:
            self.initialize_components()
        
        try:
            self.hnsw_index.load_index(filename)
            logger.info(f"Índice carregado: {filename}")
        except Exception as e:
            logger.error(f"Erro ao carregar índice: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtém status atual do sistema
        
        Returns:
            Status dos componentes
        """
        status = {
            'system_initialized': self.is_initialized,
            'components': {
                'clip_encoder': self.clip_encoder is not None,
                'hnsw_index': self.hnsw_index is not None,
                'llm': self.llm is not None,
                'metrics': self.metrics is not None
            },
            'index_built': self.hnsw_index.is_built if self.hnsw_index else False,
            'images_directory': self.images_directory,
            'images_found': len(self.clip_encoder.get_image_paths(self.images_directory)) if self.clip_encoder else 0
        }
        
        if self.hnsw_index:
            status['index_statistics'] = self.hnsw_index.get_statistics()
        
        if self.llm:
            status['llm_info'] = self.llm.get_model_info()
        
        return status
    
    def get_available_indexes(self) -> List[str]:
        """
        Lista índices salvos disponíveis
        
        Returns:
            Lista de nomes de índices
        """
        if not self.hnsw_index:
            return []
        
        return self.hnsw_index.list_saved_indexes()
    
    def test_system(self) -> Dict[str, Any]:
        """
        Testa funcionamento do sistema
        
        Returns:
            Resultado dos testes
        """
        test_results = {
            'initialization': False,
            'index_building': False,
            'search': False,
            'llm_generation': False,
            'overall_success': False,
            'errors': []
        }
        
        try:
            # Teste 1: Inicialização
            if not self.is_initialized:
                self.initialize_components()
            test_results['initialization'] = True
            
            # Teste 2: Construção do índice (se há imagens)
            image_paths = self.clip_encoder.get_image_paths(self.images_directory)
            if image_paths:
                if not self.hnsw_index.is_built:
                    self.build_index()
                test_results['index_building'] = True
                
                # Teste 3: Busca
                search_results = self.search_images("test query", k=1)
                test_results['search'] = True
                
                # Teste 4: Geração de resposta
                if search_results:
                    responses = self.generate_responses("test query", search_results[:1])
                    test_results['llm_generation'] = len(responses) > 0
            else:
                test_results['errors'].append("Nenhuma imagem encontrada para teste")
            
            # Sucesso geral
            test_results['overall_success'] = all([
                test_results['initialization'],
                test_results['index_building'] or not image_paths,
                test_results['search'] or not image_paths,
                test_results['llm_generation'] or not image_paths
            ])
            
        except Exception as e:
            test_results['errors'].append(str(e))
        
        return test_results
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """
        Obtém relatório de métricas
        
        Returns:
            Relatório de métricas ou None se desabilitado
        """
        if not self.metrics:
            return {'message': 'Métricas desabilitadas'}
        
        return self.metrics.generate_session_report()
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """
        Exporta métricas para arquivo
        
        Args:
            filename: Nome do arquivo (opcional)
            
        Returns:
            Caminho do arquivo exportado
        """
        if not self.metrics:
            raise ValueError("Métricas não estão habilitadas")
        
        return self.metrics.save_metrics_to_file(filename) 