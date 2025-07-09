#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🖼️ Sistema RAG Multimodal - Linux/Mac Launcher${NC}"
echo "============================================="
echo

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 não encontrado!${NC}"
    echo "   Instale Python 3.8+ em https://python.org"
    exit 1
fi

# Verificar se é primeira execução
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Primeira execução - Configurando ambiente...${NC}"
    echo
    
    # Criar ambiente virtual
    echo -e "${BLUE}🔧 Criando ambiente virtual...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Erro ao criar ambiente virtual${NC}"
        exit 1
    fi
    
    # Ativar ambiente virtual
    source venv/bin/activate
    
    # Executar setup
    echo -e "${BLUE}🚀 Executando setup automatizado...${NC}"
    python setup.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Erro no setup${NC}"
        exit 1
    fi
    
    echo
    echo -e "${GREEN}✅ Setup concluído!${NC}"
    echo
else
    echo -e "${BLUE}🔄 Ativando ambiente virtual...${NC}"
    source venv/bin/activate
fi

# Menu principal
show_menu() {
    echo
    echo -e "${BLUE}🎯 O que deseja fazer?${NC}"
    echo
    echo "1. 🚀 Executar interface web (Streamlit)"
    echo "2. 📊 Executar exemplo em linha de comando"
    echo "3. 🧪 Testar sistema"
    echo "4. 📁 Abrir pasta de imagens"
    echo "5. ⚙️ Configurar API do Gemini"
    echo "6. 🔄 Reinstalar dependências"
    echo "7. 🚪 Sair"
    echo
}

while true; do
    show_menu
    read -p "Digite sua escolha (1-7): " choice
    
    case $choice in
        1)
            echo
            echo -e "${BLUE}🚀 Iniciando interface web...${NC}"
            echo "   Acesse: http://localhost:8501"
            echo "   Pressione Ctrl+C para parar"
            echo
            streamlit run app.py
            ;;
        2)
            echo
            echo -e "${BLUE}📊 Executando exemplo...${NC}"
            python exemplo_uso.py
            echo
            read -p "Pressione Enter para continuar..."
            ;;
        3)
            echo
            echo -e "${BLUE}🧪 Testando sistema...${NC}"
            python -c "from rag_system import RAGMultimodalSystem; print('✅ Sistema OK!' if RAGMultimodalSystem().test_system()['overall_success'] else '❌ Sistema com problemas')"
            echo
            read -p "Pressione Enter para continuar..."
            ;;
        4)
            echo
            echo -e "${BLUE}📁 Abrindo pasta de imagens...${NC}"
            mkdir -p images
            if command -v xdg-open &> /dev/null; then
                xdg-open images  # Linux
            elif command -v open &> /dev/null; then
                open images      # Mac
            else
                echo "   Pasta: $(pwd)/images"
            fi
            echo "   Coloque suas imagens na pasta 'images/'"
            echo "   Formatos suportados: JPG, PNG, BMP, TIFF, WEBP"
            echo
            read -p "Pressione Enter para continuar..."
            ;;
        5)
            echo
            echo -e "${BLUE}⚙️ Configuração do Google Gemini${NC}"
            echo
            echo "Para usar respostas reais do LLM:"
            echo "1. Obtenha uma API key em: https://makersuite.google.com/app/apikey"
            echo "2. Edite o arquivo .env"
            echo "3. Adicione: GEMINI_API_KEY=sua_chave_aqui"
            echo
            
            if [ -f ".env" ]; then
                echo -e "${BLUE}📄 Abrindo arquivo .env...${NC}"
                if command -v nano &> /dev/null; then
                    nano .env
                elif command -v vim &> /dev/null; then
                    vim .env
                else
                    echo "   Edite o arquivo .env manualmente"
                fi
            else
                echo -e "${BLUE}📄 Criando arquivo .env...${NC}"
                echo "GEMINI_API_KEY=" > .env
                if command -v nano &> /dev/null; then
                    nano .env
                elif command -v vim &> /dev/null; then
                    vim .env
                else
                    echo "   Arquivo .env criado. Edite-o manualmente."
                fi
            fi
            echo
            read -p "Pressione Enter para continuar..."
            ;;
        6)
            echo
            echo -e "${BLUE}🔄 Reinstalando dependências...${NC}"
            pip install --upgrade pip
            pip install -r requirements.txt --force-reinstall
            echo
            echo -e "${GREEN}✅ Reinstalação concluída!${NC}"
            read -p "Pressione Enter para continuar..."
            ;;
        7)
            echo
            echo -e "${GREEN}👋 Obrigado por usar o Sistema RAG Multimodal!${NC}"
            echo
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Opção inválida!${NC}"
            ;;
    esac
done 