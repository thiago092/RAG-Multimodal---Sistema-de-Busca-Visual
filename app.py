import streamlit as st
import os
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import json
from datetime import datetime
import base64
import io

from rag_system import RAGMultimodalSystem
from config import STREAMLIT_CONFIG, IMAGES_DIR, TOP_K_RETRIEVAL

# Configuração da página
st.set_page_config(**STREAMLIT_CONFIG)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .result-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .similarity-badge {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Função para carregar sistema RAG
@st.cache_resource
def load_rag_system():
    """Carrega e inicializa o sistema RAG"""
    try:
        system = RAGMultimodalSystem()
        return system
    except Exception as e:
        st.error(f"Erro ao carregar sistema: {e}")
        return None

# Função para exibir imagem
def display_image(image_path, caption="", width=300):
    """Exibe uma imagem com tratamento de erro"""
    try:
        image = Image.open(image_path)
        st.image(image, caption=caption, width=width)
    except Exception as e:
        st.error(f"Erro ao carregar imagem {image_path}: {e}")

# Função para baixar arquivo
def get_download_link(file_path, link_text):
    """Cria link de download para arquivo"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
        return href
    except Exception as e:
        return f"Erro ao criar link: {e}"

# Título principal
st.markdown('<div class="main-header"><h1>🖼️ RAG Multimodal - Sistema de Busca Visual</h1><p>Trabalho de Estrutura de Dados - CLIP + HNSW + LLM</p></div>', unsafe_allow_html=True)

# Inicialização do sistema
system = load_rag_system()
if system is None:
    st.stop()

# Sidebar com controles
st.sidebar.title("⚙️ Controles do Sistema")

# Status do sistema
with st.sidebar:
    st.subheader("📊 Status do Sistema")
    if st.button("🔄 Atualizar Status"):
        st.cache_resource.clear()
        st.rerun()
    
    status = system.get_system_status()
    
    # Indicadores de status
    if status['system_initialized']:
        st.success("✅ Sistema Inicializado")
    else:
        st.error("❌ Sistema Não Inicializado")
    
    if status['index_built']:
        st.success("✅ Índice Construído")
        if 'index_statistics' in status:
            stats = status['index_statistics']
            st.info(f"📈 {stats['num_elements']} imagens indexadas")
    else:
        st.warning("⚠️ Índice Não Construído")
    
    st.info(f"📁 {status['images_found']} imagens encontradas")

# Seção principal
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Busca", "🏗️ Construção", "📊 Métricas", "ℹ️ Sobre"])

# Tab 1: Busca
with tab1:
    st.header("🔍 Busca por Imagens")
    
    # Verificar se sistema está pronto
    if not status['system_initialized']:
        st.warning("Sistema precisa ser inicializado primeiro. Vá para a aba 'Construção'.")
    elif not status['index_built']:
        st.warning("Índice precisa ser construído primeiro. Vá para a aba 'Construção'.")
    else:
        # Interface de busca
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "💬 Digite sua consulta:",
                placeholder="Ex: um gato preto, uma casa vermelha, pessoas caminhando..."
            )
        
        with col2:
            k = st.selectbox("🔢 Número de resultados:", [1, 3, 5, 10], index=1)
            generate_responses = st.checkbox("🤖 Gerar respostas com LLM", value=True)
        
        if st.button("🚀 Buscar", type="primary"):
            if query:
                with st.spinner("Buscando imagens..."):
                    start_time = time.time()
                    
                    # Executa consulta
                    result = system.query(query, k=k, generate_responses=generate_responses)
                    
                    search_time = time.time() - start_time
                
                if result['success']:
                    st.markdown(f'<div class="success-message">✅ Busca concluída em {search_time:.2f}s</div>', unsafe_allow_html=True)
                    
                    # Métricas da busca
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Resultados", result['num_results'])
                    with col2:
                        st.metric("💬 Respostas", result['num_responses'])
                    with col3:
                        st.metric("⏱️ Tempo", f"{search_time:.2f}s")
                    
                    # Resultados
                    if result['search_results']:
                        st.subheader("📸 Imagens Encontradas")
                        
                        for i, search_result in enumerate(result['search_results']):
                            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Exibe imagem
                                image_path = search_result['metadata']['image_path']
                                display_image(image_path, width=250)
                                
                                # Informações da imagem
                                st.markdown(f"**📁 Arquivo:** {search_result['metadata']['filename']}")
                                st.markdown(f'**🎯 Similaridade:** <span class="similarity-badge">{search_result["similarity"]:.3f}</span>', unsafe_allow_html=True)
                            
                            with col2:
                                # Resposta do LLM
                                if i < len(result['responses']):
                                    response = result['responses'][i]
                                    st.markdown("**🤖 Resposta do LLM:**")
                                    
                                    if response['success']:
                                        st.markdown(response['response'])
                                        
                                        # Informações adicionais
                                        with st.expander("ℹ️ Detalhes"):
                                            st.json({
                                                'modelo': response['model'],
                                                'rank': response['rank'],
                                                'similaridade': f"{search_result['similarity']:.3f}"
                                            })
                                    else:
                                        st.error(f"Erro na resposta: {response.get('error', 'Erro desconhecido')}")
                                else:
                                    st.info("Resposta não gerada")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Nenhuma imagem encontrada para esta consulta.")
                else:
                    st.markdown(f'<div class="error-message">❌ Erro na busca: {result.get("error", "Erro desconhecido")}</div>', unsafe_allow_html=True)
            else:
                st.warning("Digite uma consulta para buscar.")

# Tab 2: Construção
with tab2:
    st.header("🏗️ Construção do Índice")
    
    # Configurações
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Configuração de Imagens")
        
        # Diretório de imagens
        images_dir = st.text_input("Diretório das imagens:", value=IMAGES_DIR)
        
        # Botão para verificar imagens
        if st.button("🔍 Verificar Imagens"):
            if os.path.exists(images_dir):
                if system.clip_encoder is None:
                    system.initialize_components()
                
                image_paths = system.clip_encoder.get_image_paths(images_dir)
                st.success(f"✅ Encontradas {len(image_paths)} imagens")
                
                if image_paths:
                    # Mostra algumas imagens de exemplo
                    st.subheader("🖼️ Exemplo de Imagens")
                    cols = st.columns(min(3, len(image_paths)))
                    for i, path in enumerate(image_paths[:3]):
                        with cols[i]:
                            display_image(path, os.path.basename(path), width=150)
            else:
                st.error(f"❌ Diretório não encontrado: {images_dir}")
    
    with col2:
        st.subheader("⚙️ Configurações do Índice")
        
        force_rebuild = st.checkbox("🔄 Forçar reconstrução", value=False)
        
        # Informações do índice atual
        if status['index_built']:
            st.info("Índice já construído")
            if 'index_statistics' in status:
                stats = status['index_statistics']
                st.json({
                    'Elementos': stats['num_elements'],
                    'Dimensão': stats['dimension'],
                    'Espaço': stats['space'],
                    'Parâmetros': f"M={stats['M']}, ef={stats['ef_search']}"
                })
    
    # Botão para construir índice
    if st.button("🚀 Construir Índice", type="primary"):
        if not system.is_initialized:
            with st.spinner("Inicializando sistema..."):
                system.initialize_components()
        
        with st.spinner("Construindo índice... Isso pode demorar alguns minutos."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Atualiza diretório se necessário
                if images_dir != system.images_directory:
                    system.images_directory = images_dir
                
                # Constrói índice
                build_stats = system.build_index(force_rebuild=force_rebuild)
                
                progress_bar.progress(100)
                status_text.success("✅ Índice construído com sucesso!")
                
                # Exibe estatísticas
                st.subheader("📊 Estatísticas da Construção")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🖼️ Imagens Processadas", build_stats['successful_embeddings'])
                with col2:
                    st.metric("⏱️ Tempo Total", f"{build_stats['total_time']:.2f}s")
                with col3:
                    st.metric("🚀 Velocidade", f"{build_stats['embeddings_per_second']:.1f} img/s")
                
                # Gráfico de tempo
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Extração de Embeddings', 'Construção do Índice'],
                    y=[build_stats['embedding_time'], build_stats['indexing_time']],
                    marker_color=['#667eea', '#764ba2']
                ))
                fig.update_layout(
                    title="Tempo de Construção do Índice",
                    yaxis_title="Tempo (segundos)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Opção para salvar índice
                if st.button("💾 Salvar Índice"):
                    filename = f"index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    system.save_index(filename)
                    st.success(f"Índice salvo como: {filename}")
                
            except Exception as e:
                progress_bar.progress(0)
                status_text.error(f"❌ Erro ao construir índice: {e}")
                st.error(str(e))
    
    # Gerenciamento de índices salvos
    st.subheader("📦 Índices Salvos")
    
    saved_indexes = system.get_available_indexes()
    if saved_indexes:
        selected_index = st.selectbox("Selecione um índice:", saved_indexes)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Carregar Índice"):
                try:
                    system.load_index(selected_index)
                    st.success(f"✅ Índice carregado: {selected_index}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro ao carregar índice: {e}")
        
        with col2:
            if st.button("🗑️ Deletar Índice"):
                try:
                    system.hnsw_index.delete_index(selected_index)
                    st.success(f"✅ Índice deletado: {selected_index}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro ao deletar índice: {e}")
    else:
        st.info("Nenhum índice salvo encontrado.")

# Tab 3: Métricas
with tab3:
    st.header("📊 Métricas e Análise")
    
    # Relatório de métricas
    metrics_report = system.get_metrics_report()
    
    if 'message' in metrics_report:
        st.info(metrics_report['message'])
    else:
        # Resumo da sessão
        st.subheader("📈 Resumo da Sessão")
        
        session_info = metrics_report.get('session_info', {})
        perf_metrics = metrics_report.get('performance_metrics', {})
        quality_metrics = metrics_report.get('quality_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔍 Consultas Realizadas",
                session_info.get('total_queries', 0)
            )
        
        with col2:
            st.metric(
                "⏱️ Tempo Médio",
                f"{perf_metrics.get('avg_total_time', 0):.2f}s"
            )
        
        with col3:
            st.metric(
                "🎯 Similaridade Média",
                f"{quality_metrics.get('avg_similarity', 0):.3f}"
            )
        
        with col4:
            st.metric(
                "✅ Taxa de Sucesso",
                f"{quality_metrics.get('overall_success_rate', 0):.1%}"
            )
        
        # Gráficos de desempenho
        if session_info.get('total_queries', 0) > 0:
            st.subheader("📊 Análise de Desempenho")
            
            # Gráfico de tempo de resposta
            if system.metrics and system.metrics.metrics_data:
                df = pd.DataFrame(system.metrics.metrics_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.line(
                        df, 
                        y='total_time',
                        title="Tempo de Resposta por Consulta",
                        labels={'index': 'Consulta', 'total_time': 'Tempo (s)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        df,
                        x='avg_similarity',
                        title="Distribuição de Similaridade",
                        labels={'avg_similarity': 'Similaridade', 'count': 'Frequência'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Análise de consultas
                st.subheader("🔤 Análise de Consultas")
                
                query_analysis = metrics_report.get('query_analysis', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Palavras Mais Comuns:**")
                    common_words = query_analysis.get('most_common_words', [])
                    for word_info in common_words[:5]:
                        st.markdown(f"- {word_info['word']}: {word_info['count']} vezes")
                
                with col2:
                    st.markdown("**Estatísticas de Consulta:**")
                    query_stats = query_analysis.get('query_length_stats', {})
                    st.markdown(f"- Comprimento médio: {query_stats.get('avg_length', 0):.1f} caracteres")
                    st.markdown(f"- Maior consulta: {query_stats.get('max_length', 0)} caracteres")
                    st.markdown(f"- Menor consulta: {query_stats.get('min_length', 0)} caracteres")
        
        # Exportar métricas
        st.subheader("📥 Exportar Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Exportar Métricas"):
                try:
                    filepath = system.export_metrics()
                    st.success(f"✅ Métricas exportadas: {os.path.basename(filepath)}")
                    
                    # Link para download
                    st.markdown(get_download_link(filepath, "📥 Baixar Arquivo"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Erro ao exportar: {e}")
        
        with col2:
            if st.button("📊 Gerar Gráficos"):
                try:
                    if system.metrics:
                        plot_paths = system.metrics.generate_performance_plots()
                        st.success(f"✅ {len(plot_paths)} gráficos gerados")
                        
                        for path in plot_paths:
                            st.markdown(get_download_link(path, f"📥 Baixar {os.path.basename(path)}"), unsafe_allow_html=True)
                    else:
                        st.error("Métricas não disponíveis")
                except Exception as e:
                    st.error(f"❌ Erro ao gerar gráficos: {e}")

# Tab 4: Sobre
with tab4:
    st.header("ℹ️ Sobre o Sistema")
    
    st.markdown("""
    ## 🎯 Objetivo do Projeto
    
    Este sistema implementa um **RAG (Retrieval-Augmented Generation) Multimodal** para recuperação de imagens baseada em consultas textuais, desenvolvido como trabalho para a disciplina de Estrutura de Dados.
    
    ## 🏗️ Arquitetura do Sistema
    
    O sistema integra três componentes principais:
    
    ### 1. 🖼️ CLIP (Contrastive Language-Image Pre-training)
    - Converte imagens e texto em embeddings no mesmo espaço vetorial
    - Permite busca semântica cross-modal
    - Modelo usado: ViT-B/32
    
    ### 2. 🔍 HNSW (Hierarchical Navigable Small World)
    - Estrutura de dados para busca vetorial aproximada
    - Indexação eficiente de embeddings de alta dimensão
    - Busca rápida com complexidade logarítmica
    
    ### 3. 🤖 LLM Multimodal (Google Gemini)
    - Gera respostas textuais baseadas nas imagens recuperadas
    - Análise contextual das imagens
    - Respostas detalhadas e precisas
    
    ## 📊 Características Técnicas
    
    - **Linguagem**: Python
    - **Interface**: Streamlit
    - **Métricas**: Tempo de resposta, similaridade, taxa de sucesso
    - **Escalabilidade**: Suporta milhares de imagens
    - **Flexibilidade**: Configurações ajustáveis
    
    ## 🚀 Funcionalidades
    
    - ✅ Indexação automática de imagens
    - ✅ Busca semântica por texto
    - ✅ Geração de respostas contextuais
    - ✅ Análise de métricas de desempenho
    - ✅ Exportação de dados para apresentação
    - ✅ Interface web intuitiva
    
    ## 📈 Métricas Coletadas
    
    - **Desempenho**: Tempo total, tempo por componente
    - **Qualidade**: Similaridade, taxa de sucesso
    - **Uso**: Consultas realizadas, padrões de busca
    - **Análise**: Distribuições, tendências, comparações
    
    ## 🔧 Configuração
    
    Para usar o LLM real (Google Gemini), configure a variável `GEMINI_API_KEY` no arquivo `.env`.
    Sem a chave da API, o sistema funciona com respostas simuladas.
    """)
    
    # Informações do sistema
    st.subheader("🔧 Informações do Sistema")
    
    system_info = {
        'Status': 'Inicializado' if status['system_initialized'] else 'Não Inicializado',
        'Índice': 'Construído' if status['index_built'] else 'Não Construído',
        'Imagens': status['images_found'],
        'Diretório': status['images_directory'],
        'Componentes': {
            'CLIP': '✅' if status['components']['clip_encoder'] else '❌',
            'HNSW': '✅' if status['components']['hnsw_index'] else '❌',
            'LLM': '✅' if status['components']['llm'] else '❌',
            'Métricas': '✅' if status['components']['metrics'] else '❌'
        }
    }
    
    st.json(system_info)
    
    # Teste do sistema
    st.subheader("🧪 Teste do Sistema")
    
    if st.button("🚀 Executar Teste Completo"):
        with st.spinner("Executando testes..."):
            test_results = system.test_system()
        
        st.markdown("**Resultados dos Testes:**")
        
        for test_name, result in test_results.items():
            if test_name != 'errors':
                icon = "✅" if result else "❌"
                st.markdown(f"- {icon} {test_name.replace('_', ' ').title()}")
        
        if test_results.get('errors'):
            st.error("Erros encontrados:")
            for error in test_results['errors']:
                st.markdown(f"- {error}")
        
        if test_results.get('overall_success'):
            st.success("🎉 Sistema funcionando corretamente!")
        else:
            st.warning("⚠️ Sistema apresenta problemas. Verifique os erros acima.")

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>💡 <strong>RAG Multimodal System</strong> - Trabalho de Estrutura de Dados</p>
    <p>Desenvolvido com ❤️ usando Python, Streamlit, CLIP, HNSW e Google Gemini</p>
</div>
""", unsafe_allow_html=True) 