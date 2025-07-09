#!/usr/bin/env python3
"""
Setup e Configuração Automatizada do Sistema RAG Multimodal
===========================================================

Este script automatiza a instalação e configuração inicial do sistema.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Imprime cabeçalho do script"""
    print("🖼️ Sistema RAG Multimodal - Setup Automatizado")
    print("=" * 60)
    print("Trabalho de Estrutura de Dados")
    print("=" * 60)

def check_python_version():
    """Verifica versão do Python"""
    print("🐍 Verificando versão do Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário")
        print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} OK")
    return True

def create_directories():
    """Cria diretórios necessários"""
    print("\n📁 Criando diretórios...")
    
    directories = [
        "images",
        "index", 
        "cache",
        "results"
    ]
    
    for dir_name in directories:
        path = Path(dir_name)
        path.mkdir(exist_ok=True)
        print(f"   ✅ {dir_name}/")

def install_requirements():
    """Instala dependências"""
    print("\n📦 Instalando dependências...")
    
    try:
        # Atualizar pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Instalar requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependências instaladas com sucesso!")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def check_cuda():
    """Verifica disponibilidade de CUDA"""
    print("\n🔍 Verificando CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA disponível!")
            print(f"   Dispositivos: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   - {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("ℹ️ CUDA não disponível (usando CPU)")
            return False
    except ImportError:
        print("⚠️ PyTorch não instalado ainda")
        return False

def create_env_file():
    """Cria arquivo .env de exemplo"""
    print("\n⚙️ Criando arquivo de configuração...")
    
    env_content = """# Configuração do Sistema RAG Multimodal
# =====================================

# Google Gemini API Key (opcional)
# Obtenha em: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=

# Usar GPU se disponível
USE_CUDA=false

# Configurações de desenvolvimento
DEBUG=false
LOG_LEVEL=INFO

# Configurações de interface
STREAMLIT_THEME=light
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_content)
        print("✅ Arquivo .env criado")
        print("   Configure GEMINI_API_KEY para usar LLM real")
    else:
        print("ℹ️ Arquivo .env já existe")

def test_imports():
    """Testa importações principais"""
    print("\n🧪 Testando importações...")
    
    modules = [
        ("streamlit", "Streamlit"),
        ("torch", "PyTorch"),
        ("clip", "CLIP"),
        ("hnswlib", "HNSW"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly")
    ]
    
    failed = []
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️ Módulos com problemas: {', '.join(failed)}")
        return False
    else:
        print("\n✅ Todas as importações OK!")
        return True

def create_sample_images():
    """Cria imagens de exemplo se necessário"""
    print("\n🎨 Verificando imagens de exemplo...")
    
    images_dir = Path("images")
    existing_images = list(images_dir.glob("*"))
    
    if existing_images:
        print(f"✅ Encontradas {len(existing_images)} imagens")
        return
    
    print("Criando imagens de exemplo...")
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random
        
        # Cores e formas
        cores = ["red", "blue", "green", "yellow", "purple", "orange"]
        formas = ["circle", "square", "triangle"]
        
        for i in range(6):
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
        
        print(f"✅ Criadas 6 imagens de exemplo")
    
    except Exception as e:
        print(f"⚠️ Erro ao criar imagens: {e}")

def test_system():
    """Testa o sistema básico"""
    print("\n🚀 Testando sistema básico...")
    
    try:
        # Importar sistema
        from rag_system import RAGMultimodalSystem
        
        # Criar sistema
        system = RAGMultimodalSystem()
        
        # Testar inicialização
        system.initialize_components()
        
        # Verificar status
        status = system.get_system_status()
        
        print("✅ Sistema inicializado com sucesso!")
        print(f"   - Imagens encontradas: {status['images_found']}")
        print(f"   - Componentes OK: {sum(status['components'].values())}/4")
        
        return True
    
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def print_next_steps():
    """Imprime próximos passos"""
    print("\n🎯 Próximos Passos")
    print("=" * 30)
    print()
    print("1. 🚀 Executar a aplicação:")
    print("   streamlit run app.py")
    print()
    print("2. 📱 Acessar no navegador:")
    print("   http://localhost:8501")
    print()
    print("3. 🖼️ Adicionar suas imagens:")
    print("   - Coloque imagens na pasta 'images/'")
    print("   - Formatos: JPG, PNG, BMP, TIFF, WEBP")
    print()
    print("4. 🤖 Configurar LLM (opcional):")
    print("   - Edite o arquivo .env")
    print("   - Adicione sua GEMINI_API_KEY")
    print()
    print("5. 📊 Executar exemplo:")
    print("   python exemplo_uso.py")
    print()

def main():
    """Função principal"""
    print_header()
    
    # Verificações
    if not check_python_version():
        sys.exit(1)
    
    # Criar diretórios
    create_directories()
    
    # Instalar dependências
    if not install_requirements():
        print("\n❌ Instalação falhou!")
        sys.exit(1)
    
    # Verificar CUDA
    check_cuda()
    
    # Criar arquivo .env
    create_env_file()
    
    # Testar importações
    if not test_imports():
        print("\n⚠️ Algumas importações falharam, mas o sistema pode funcionar")
    
    # Criar imagens de exemplo
    create_sample_images()
    
    # Testar sistema
    if test_system():
        print("\n🎉 Setup concluído com sucesso!")
    else:
        print("\n⚠️ Setup concluído com avisos")
    
    # Próximos passos
    print_next_steps()

if __name__ == "__main__":
    main() 