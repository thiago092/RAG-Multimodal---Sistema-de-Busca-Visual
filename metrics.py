import time
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from config import RESULTS_DIR, METRICS_ENABLED

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGMetrics:
    """
    Classe para coletar e analisar métricas do sistema RAG
    """
    
    def __init__(self):
        """Inicializa sistema de métricas"""
        self.metrics_data = []
        self.session_start = datetime.now()
        
        # Criar diretório de resultados
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def start_query_timer(self) -> float:
        """
        Inicia timer para consulta
        
        Returns:
            Timestamp de início
        """
        return time.time()
    
    def record_query_metrics(self, query: str, results: List[Dict[str, Any]], 
                           responses: List[Dict[str, Any]], 
                           start_time: float, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Registra métricas de uma consulta
        
        Args:
            query: Consulta realizada
            results: Resultados da busca
            responses: Respostas geradas
            start_time: Timestamp de início
            metadata: Metadados adicionais
            
        Returns:
            Dicionário com métricas da consulta
        """
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calcula métricas básicas
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'total_time': total_time,
            'num_results': len(results),
            'num_responses': len(responses),
            'avg_similarity': np.mean([r.get('similarity', 0) for r in results]) if results else 0,
            'max_similarity': max([r.get('similarity', 0) for r in results]) if results else 0,
            'min_similarity': min([r.get('similarity', 0) for r in results]) if results else 0,
            'successful_responses': sum(1 for r in responses if r.get('success', False)),
            'failed_responses': sum(1 for r in responses if not r.get('success', False)),
        }
        
        # Adiciona métricas de tempo detalhadas
        if metadata:
            metrics.update({
                'clip_encoding_time': metadata.get('clip_encoding_time', 0),
                'hnsw_search_time': metadata.get('hnsw_search_time', 0),
                'llm_generation_time': metadata.get('llm_generation_time', 0),
            })
        
        # Adiciona métricas de qualidade
        metrics.update(self._calculate_quality_metrics(results, responses))
        
        # Salva métricas
        if METRICS_ENABLED:
            self.metrics_data.append(metrics)
            logger.info(f"Métricas registradas: {query[:50]}... ({total_time:.2f}s)")
        
        return metrics
    
    def _calculate_quality_metrics(self, results: List[Dict[str, Any]], 
                                 responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula métricas de qualidade
        
        Args:
            results: Resultados da busca
            responses: Respostas geradas
            
        Returns:
            Métricas de qualidade
        """
        quality_metrics = {}
        
        if results:
            # Diversidade de similaridades
            similarities = [r.get('similarity', 0) for r in results]
            quality_metrics['similarity_std'] = np.std(similarities)
            quality_metrics['similarity_range'] = max(similarities) - min(similarities)
            
            # Distribuição de resultados
            quality_metrics['top_1_similarity'] = similarities[0] if similarities else 0
            quality_metrics['top_3_avg_similarity'] = np.mean(similarities[:3]) if len(similarities) >= 3 else np.mean(similarities)
        
        if responses:
            # Métricas de resposta
            response_lengths = [len(r.get('response', '')) for r in responses]
            quality_metrics['avg_response_length'] = np.mean(response_lengths)
            quality_metrics['response_length_std'] = np.std(response_lengths)
            
            # Taxa de sucesso
            success_rate = sum(1 for r in responses if r.get('success', False)) / len(responses)
            quality_metrics['success_rate'] = success_rate
        
        return quality_metrics
    
    def generate_session_report(self) -> Dict[str, Any]:
        """
        Gera relatório da sessão atual
        
        Returns:
            Relatório da sessão
        """
        if not self.metrics_data:
            return {'message': 'Nenhuma métrica coletada ainda'}
        
        df = pd.DataFrame(self.metrics_data)
        
        report = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'duration': (datetime.now() - self.session_start).total_seconds(),
                'total_queries': len(self.metrics_data)
            },
            'performance_metrics': {
                'avg_total_time': float(df['total_time'].mean()),
                'max_total_time': float(df['total_time'].max()),
                'min_total_time': float(df['total_time'].min()),
                'std_total_time': float(df['total_time'].std()),
                'avg_results_per_query': float(df['num_results'].mean()),
                'total_results_retrieved': int(df['num_results'].sum())
            },
            'quality_metrics': {
                'avg_similarity': float(df['avg_similarity'].mean()),
                'max_similarity': float(df['max_similarity'].max()),
                'overall_success_rate': float(df['successful_responses'].sum() / df['num_responses'].sum()) if df['num_responses'].sum() > 0 else 0,
                'avg_response_length': float(df['avg_response_length'].mean()) if 'avg_response_length' in df.columns else 0
            },
            'query_analysis': {
                'most_common_words': self._analyze_query_words(df['query'].tolist()),
                'query_length_stats': {
                    'avg_length': float(df['query'].str.len().mean()),
                    'max_length': int(df['query'].str.len().max()),
                    'min_length': int(df['query'].str.len().min())
                }
            }
        }
        
        return report
    
    def _analyze_query_words(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Analisa palavras mais comuns nas consultas
        
        Args:
            queries: Lista de consultas
            
        Returns:
            Lista de palavras mais comuns
        """
        from collections import Counter
        
        # Extrai palavras (simples)
        words = []
        for query in queries:
            words.extend(query.lower().split())
        
        # Remove palavras muito comuns
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'what', 'where', 'when', 'why', 'how', 'this', 'that', 'these', 'those'}
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Conta palavras
        word_counts = Counter(words)
        
        return [{'word': word, 'count': count} for word, count in word_counts.most_common(10)]
    
    def save_metrics_to_file(self, filename: Optional[str] = None) -> str:
        """
        Salva métricas em arquivo
        
        Args:
            filename: Nome do arquivo (opcional)
            
        Returns:
            Caminho do arquivo salvo
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_metrics_{timestamp}.json"
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        # Salva dados e relatório
        data_to_save = {
            'metrics_data': self.metrics_data,
            'session_report': self.generate_session_report(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Métricas salvas em: {filepath}")
        return filepath
    
    def generate_performance_plots(self) -> List[str]:
        """
        Gera gráficos de desempenho
        
        Returns:
            Lista de caminhos dos gráficos gerados
        """
        if not self.metrics_data:
            return []
        
        df = pd.DataFrame(self.metrics_data)
        plot_paths = []
        
        # Configuração de estilo
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Gráfico de tempo de resposta
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(df['total_time'], marker='o')
        plt.title('Tempo de Resposta por Consulta')
        plt.xlabel('Consulta')
        plt.ylabel('Tempo (segundos)')
        plt.grid(True, alpha=0.3)
        
        # 2. Distribuição de similaridades
        plt.subplot(1, 2, 2)
        plt.hist(df['avg_similarity'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribuição de Similaridade Média')
        plt.xlabel('Similaridade')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, 'performance_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # 3. Gráfico interativo com Plotly
        fig = go.Figure()
        
        # Tempo vs Similaridade
        fig.add_trace(go.Scatter(
            x=df['total_time'],
            y=df['avg_similarity'],
            mode='markers',
            marker=dict(
                size=df['num_results']*3,
                color=df['success_rate'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Taxa de Sucesso")
            ),
            text=df['query'].str[:50] + '...',
            hovertemplate='<b>Consulta:</b> %{text}<br>' +
                         '<b>Tempo:</b> %{x:.2f}s<br>' +
                         '<b>Similaridade:</b> %{y:.3f}<br>' +
                         '<b>Resultados:</b> %{marker.size}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Desempenho do Sistema RAG',
            xaxis_title='Tempo de Resposta (segundos)',
            yaxis_title='Similaridade Média',
            hovermode='closest'
        )
        
        interactive_path = os.path.join(RESULTS_DIR, 'interactive_performance.html')
        fig.write_html(interactive_path)
        plot_paths.append(interactive_path)
        
        logger.info(f"Gráficos gerados: {len(plot_paths)} arquivos")
        return plot_paths
    
    def get_benchmark_comparison(self) -> Dict[str, Any]:
        """
        Compara desempenho com benchmarks típicos
        
        Returns:
            Comparação com benchmarks
        """
        if not self.metrics_data:
            return {}
        
        df = pd.DataFrame(self.metrics_data)
        avg_time = df['total_time'].mean()
        avg_similarity = df['avg_similarity'].mean()
        success_rate = df['successful_responses'].sum() / df['num_responses'].sum() if df['num_responses'].sum() > 0 else 0
        
        # Benchmarks típicos (valores aproximados)
        benchmarks = {
            'response_time': {
                'excellent': 1.0,
                'good': 3.0,
                'acceptable': 5.0,
                'poor': 10.0
            },
            'similarity_score': {
                'excellent': 0.9,
                'good': 0.8,
                'acceptable': 0.7,
                'poor': 0.6
            },
            'success_rate': {
                'excellent': 0.95,
                'good': 0.90,
                'acceptable': 0.85,
                'poor': 0.80
            }
        }
        
        def get_rating(value, benchmark):
            if value <= benchmark['excellent']:
                return 'excellent'
            elif value <= benchmark['good']:
                return 'good'
            elif value <= benchmark['acceptable']:
                return 'acceptable'
            else:
                return 'poor'
        
        def get_similarity_rating(value, benchmark):
            if value >= benchmark['excellent']:
                return 'excellent'
            elif value >= benchmark['good']:
                return 'good'
            elif value >= benchmark['acceptable']:
                return 'acceptable'
            else:
                return 'poor'
        
        comparison = {
            'current_performance': {
                'avg_response_time': avg_time,
                'avg_similarity': avg_similarity,
                'success_rate': success_rate
            },
            'ratings': {
                'response_time': get_rating(avg_time, benchmarks['response_time']),
                'similarity_score': get_similarity_rating(avg_similarity, benchmarks['similarity_score']),
                'success_rate': get_similarity_rating(success_rate, benchmarks['success_rate'])
            },
            'recommendations': self._generate_recommendations(avg_time, avg_similarity, success_rate)
        }
        
        return comparison
    
    def _generate_recommendations(self, avg_time: float, avg_similarity: float, 
                                success_rate: float) -> List[str]:
        """
        Gera recomendações baseadas nas métricas
        
        Args:
            avg_time: Tempo médio de resposta
            avg_similarity: Similaridade média
            success_rate: Taxa de sucesso
            
        Returns:
            Lista de recomendações
        """
        recommendations = []
        
        if avg_time > 5.0:
            recommendations.append("Considere otimizar o tempo de resposta usando GPU ou índices menores")
        
        if avg_similarity < 0.7:
            recommendations.append("Melhore a qualidade dos embeddings ou ajuste os parâmetros do CLIP")
        
        if success_rate < 0.9:
            recommendations.append("Verifique a configuração do LLM ou trate melhor os casos de erro")
        
        if not recommendations:
            recommendations.append("Sistema funcionando bem! Continue monitorando o desempenho")
        
        return recommendations
    
    def export_for_presentation(self) -> str:
        """
        Exporta métricas formatadas para apresentação
        
        Returns:
            Caminho do arquivo de apresentação
        """
        report = self.generate_session_report()
        benchmark = self.get_benchmark_comparison()
        
        presentation_data = {
            'titulo': 'Relatório de Desempenho - Sistema RAG Multimodal',
            'resumo_executivo': {
                'total_consultas': report['session_info']['total_queries'],
                'tempo_medio_resposta': f"{report['performance_metrics']['avg_total_time']:.2f}s",
                'similaridade_media': f"{report['quality_metrics']['avg_similarity']:.3f}",
                'taxa_sucesso': f"{report['quality_metrics']['overall_success_rate']:.1%}",
                'avaliacao_geral': benchmark.get('ratings', {})
            },
            'metricas_detalhadas': report,
            'comparacao_benchmark': benchmark,
            'recomendacoes': benchmark.get('recommendations', [])
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(RESULTS_DIR, f"apresentacao_metricas_{timestamp}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(presentation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dados para apresentação salvos em: {filepath}")
        return filepath 