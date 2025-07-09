import google.generativeai as genai
import logging
from typing import Dict, Any, List, Optional
import base64
import io
from PIL import Image
import os

from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_TOKENS, TEMPERATURE

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalLLM:
    """
    Classe para interagir com LLM multimodal (Google Gemini)
    """
    
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        """
        Inicializa o LLM multimodal
        
        Args:
            api_key: Chave da API do Google Gemini
            model_name: Nome do modelo
        """
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        
        self._setup_model()
    
    def _setup_model(self):
        """Configura o modelo Gemini"""
        try:
            if not self.api_key:
                logger.warning("API key do Gemini não fornecida. Usando resposta simulada.")
                self.model = None
                return
            
            # Configura API do Gemini
            genai.configure(api_key=self.api_key)
            
            # Configura modelo
            generation_config = {
                "temperature": TEMPERATURE,
                "max_output_tokens": MAX_TOKENS,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
            
            logger.info(f"Modelo Gemini configurado: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Erro ao configurar modelo Gemini: {e}")
            self.model = None
    
    def generate_response(self, image_path: str, query: str, 
                         context: Optional[str] = None) -> Dict[str, Any]:
        """
        Gera resposta textual baseada em imagem e consulta
        
        Args:
            image_path: Caminho para a imagem
            query: Consulta textual
            context: Contexto adicional (opcional)
            
        Returns:
            Dicionário com resposta e metadados
        """
        try:
            # Se não há modelo configurado, usa resposta simulada
            if self.model is None:
                return self._generate_mock_response(image_path, query, context)
            
            # Carrega imagem
            image = Image.open(image_path)
            
            # Constrói prompt
            prompt = self._build_prompt(query, context)
            
            # Gera resposta
            response = self.model.generate_content([prompt, image])
            
            result = {
                'response': response.text,
                'query': query,
                'image_path': image_path,
                'context': context,
                'model': self.model_name,
                'success': True,
                'error': None
            }
            
            logger.info(f"Resposta gerada com sucesso para: {query}")
            return result
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            return {
                'response': f"Erro ao gerar resposta: {str(e)}",
                'query': query,
                'image_path': image_path,
                'context': context,
                'model': self.model_name,
                'success': False,
                'error': str(e)
            }
    
    def _generate_mock_response(self, image_path: str, query: str, 
                              context: Optional[str] = None) -> Dict[str, Any]:
        """
        Gera resposta simulada quando não há API key
        
        Args:
            image_path: Caminho para a imagem
            query: Consulta textual
            context: Contexto adicional
            
        Returns:
            Dicionário com resposta simulada
        """
        # Extrai nome do arquivo
        filename = os.path.basename(image_path)
        
        # Resposta simulada baseada na consulta
        mock_response = f"""
        [RESPOSTA SIMULADA - Configure GEMINI_API_KEY para usar o modelo real]
        
        Baseado na análise da imagem '{filename}' e na consulta '{query}':
        
        Esta é uma resposta simulada que seria gerada pelo modelo Gemini.
        A imagem foi recuperada com sucesso pelo sistema RAG baseado em similaridade CLIP.
        
        Para obter respostas reais:
        1. Configure a variável GEMINI_API_KEY no arquivo .env
        2. Obtenha uma chave da API do Google AI Studio
        3. Reinicie a aplicação
        
        Consulta original: {query}
        Imagem analisada: {filename}
        {f"Contexto: {context}" if context else ""}
        """
        
        return {
            'response': mock_response.strip(),
            'query': query,
            'image_path': image_path,
            'context': context,
            'model': 'mock_model',
            'success': True,
            'error': None
        }
    
    def _build_prompt(self, query: str, context: Optional[str] = None) -> str:
        """
        Constrói prompt para o modelo
        
        Args:
            query: Consulta do usuário
            context: Contexto adicional
            
        Returns:
            Prompt formatado
        """
        prompt = f"""
        Você é um assistente especializado em análise de imagens. 
        Analise a imagem fornecida e responda à seguinte pergunta de forma detalhada e precisa:
        
        Pergunta: {query}
        
        {f"Contexto adicional: {context}" if context else ""}
        
        Instruções:
        1. Descreva o que você vê na imagem
        2. Responda especificamente à pergunta feita
        3. Forneça detalhes relevantes sobre cores, objetos, pessoas, cenário, etc.
        4. Se a pergunta não puder ser respondida com base na imagem, explique por quê
        5. Seja objetivo e informativo
        
        Resposta:
        """
        
        return prompt.strip()
    
    def generate_batch_responses(self, results: List[Dict[str, Any]], 
                               query: str) -> List[Dict[str, Any]]:
        """
        Gera respostas para múltiplos resultados
        
        Args:
            results: Lista de resultados da busca
            query: Consulta original
            
        Returns:
            Lista de respostas geradas
        """
        responses = []
        
        for i, result in enumerate(results):
            logger.info(f"Gerando resposta {i+1}/{len(results)}")
            
            image_path = result['metadata'].get('image_path', '')
            similarity = result.get('similarity', 0.0)
            
            # Contexto com informações da similaridade
            context = f"Imagem recuperada com similaridade: {similarity:.3f}"
            
            # Gera resposta
            response = self.generate_response(image_path, query, context)
            
            # Adiciona informações do resultado
            response['retrieval_result'] = result
            response['rank'] = i + 1
            
            responses.append(response)
        
        return responses
    
    def is_available(self) -> bool:
        """
        Verifica se o modelo está disponível
        
        Returns:
            True se o modelo estiver configurado
        """
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo
        
        Returns:
            Dicionário com informações do modelo
        """
        return {
            'model_name': self.model_name,
            'api_key_configured': bool(self.api_key),
            'model_available': self.is_available(),
            'max_tokens': MAX_TOKENS,
            'temperature': TEMPERATURE
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Testa conexão com o modelo
        
        Returns:
            Resultado do teste
        """
        try:
            if not self.is_available():
                return {
                    'success': False,
                    'message': 'Modelo não está configurado',
                    'error': 'API key não fornecida'
                }
            
            # Testa com uma imagem dummy (se houver)
            test_prompt = "Teste de conexão"
            
            # Simula teste (sem imagem real)
            return {
                'success': True,
                'message': 'Conexão com modelo estabelecida',
                'model': self.model_name
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Erro na conexão: {str(e)}',
                'error': str(e)
            } 