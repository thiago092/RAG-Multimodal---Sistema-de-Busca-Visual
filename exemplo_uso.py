#!/usr/bin/env python3
"""
Exemplo de uso do Sistema RAG Multimodal
========================================

Este script demonstra como usar o sistema RAG multimodal
programaticamente, sem a interface web.
"""

import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Importar sistema RAG
from rag_system import RAGMultimodalSystem

def main():
    """Função principal de demonstração"""
    print("🖼️ Sistema RAG Multimodal - Exemplo de Uso")
    print("=" * 50)
    
    # 1. Criar sistema
    print("\n1. Inicializando sistema...")
    system = RAGMultimodalSystem(images_directory="images")
    
    # 2. Verificar se há imagens
    print("\n2. Verificando imagens...")
    if not system.is_initialized:
        system.initialize_components()
    
    image_paths = system.clip_encoder.get_image_paths(system.images_directory)
    if not image_paths:
        print("❌ Nenhuma imagem encontrada!")
        print(f"   Coloque algumas imagens na pasta '{system.images_directory}'")
        print("   Formatos suportados: JPG, JPEG, PNG, BMP, TIFF, WEBP")
        return
    
    print(f"✅ Encontradas {len(image_paths)} imagens")
    
    # 3. Construir índice
    print("\n3. Construindo índice...")
    try:
        build_stats = system.build_index()
        print(f"✅ Índice construído com sucesso!")
        print(f"   - Imagens processadas: {build_stats['successful_embeddings']}")
        print(f"   - Tempo total: {build_stats['total_time']:.2f}s")
        print(f"   - Velocidade: {build_stats['embeddings_per_second']:.1f} img/s")
    except Exception as e:
        print(f"❌ Erro ao construir índice: {e}")
        return
    
    # 4. Exemplos de consultas
    consultas_exemplo = [
        "um gato",
        "pessoas",
        "casa",
        "carro",
        "natureza",
        "comida"
    ]
    
    print("\n4. Testando consultas...")
    
    for i, consulta in enumerate(consultas_exemplo, 1):
        print(f"\n--- Consulta {i}: '{consulta}' ---")
        
        try:
            # Realizar busca
            resultado = system.query(consulta, k=2, generate_responses=True)
            
            if resultado['success']:
                print(f"✅ Encontrados {resultado['num_results']} resultados")
                print(f"   Tempo: {resultado['timing']['total_time']:.2f}s")
                
                # Mostrar resultados
                for j, res in enumerate(resultado['search_results'], 1):
                    filename = os.path.basename(res['metadata']['image_path'])
                    print(f"   {j}. {filename} (similaridade: {res['similarity']:.3f})")
                
                # Mostrar resposta do LLM (primeira imagem)
                if resultado['responses']:
                    response = resultado['responses'][0]
                    if response['success']:
                        print(f"   🤖 Resposta LLM: {response['response'][:100]}...")
                    else:
                        print(f"   ❌ Erro no LLM: {response.get('error', 'Desconhecido')}")
            else:
                print(f"❌ Erro na consulta: {resultado.get('error', 'Desconhecido')}")
        
        except Exception as e:
            print(f"❌ Erro na consulta: {e}")
    
    # 5. Métricas
    print("\n5. Métricas da sessão...")
    try:
        relatorio = system.get_metrics_report()
        
        if 'message' not in relatorio:
            print("📊 Resumo das métricas:")
            session_info = relatorio.get('session_info', {})
            perf_metrics = relatorio.get('performance_metrics', {})
            quality_metrics = relatorio.get('quality_metrics', {})
            
            print(f"   - Consultas realizadas: {session_info.get('total_queries', 0)}")
            print(f"   - Tempo médio: {perf_metrics.get('avg_total_time', 0):.2f}s")
            print(f"   - Similaridade média: {quality_metrics.get('avg_similarity', 0):.3f}")
            print(f"   - Taxa de sucesso: {quality_metrics.get('overall_success_rate', 0):.1%}")
            
            # Exportar métricas
            filepath = system.export_metrics("exemplo_metricas.json")
            print(f"   - Métricas exportadas: {os.path.basename(filepath)}")
        else:
            print("ℹ️ Métricas não disponíveis")
    
    except Exception as e:
        print(f"❌ Erro ao gerar métricas: {e}")
    
    # 6. Salvar índice
    print("\n6. Salvando índice...")
    try:
        system.save_index("exemplo_index")
        print("✅ Índice salvo com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao salvar índice: {e}")
    
    print("\n🎉 Demonstração concluída!")
    print("\nPara usar a interface web, execute:")
    print("   streamlit run app.py")

def criar_imagens_exemplo():
    """Cria algumas imagens de exemplo se não existirem"""
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)
    
    # Verificar se já existem imagens
    existing_images = list(images_dir.glob("*.png"))
    if existing_images:
        print(f"✅ Já existem {len(existing_images)} imagens")
        return
    
    print("🎨 Criando imagens de exemplo...")
    
    # Cores e formas
    cores = ["red", "blue", "green", "yellow", "purple", "orange"]
    formas = ["circle", "square", "triangle"]
    
    for i in range(10):
        # Criar imagem
        img = Image.new('RGB', (300, 300), 'white')
        draw = ImageDraw.Draw(img)
        
        # Desenhar forma aleatória
        cor = random.choice(cores)
        forma = random.choice(formas)
        
        if forma == "circle":
            draw.ellipse([50, 50, 250, 250], fill=cor)
        elif forma == "square":
            draw.rectangle([50, 50, 250, 250], fill=cor)
        else:  # triangle
            draw.polygon([(150, 50), (50, 250), (250, 250)], fill=cor)
        
        # Adicionar texto
        try:
            font = ImageFont.load_default()
            draw.text((10, 10), f"{cor} {forma}", fill="black", font=font)
        except:
            draw.text((10, 10), f"{cor} {forma}", fill="black")
        
        # Salvar
        filename = f"exemplo_{i+1:02d}_{cor}_{forma}.png"
        img.save(images_dir / filename)
    
    print(f"✅ Criadas 10 imagens de exemplo em {images_dir}")

if __name__ == "__main__":
    # Verificar se existem imagens, se não, criar exemplos
    if not os.path.exists("images") or not os.listdir("images"):
        print("📁 Pasta de imagens vazia. Criando exemplos...")
        criar_imagens_exemplo()
    
    # Executar demonstração
    main() 