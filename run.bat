@echo off
chcp 65001 >nul
echo 🖼️ Sistema RAG Multimodal - Windows Launcher
echo =============================================
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python não encontrado!
    echo    Instale Python 3.8+ em https://python.org
    pause
    exit /b 1
)

REM Verificar se é primeira execução
if not exist "venv" (
    echo 📦 Primeira execução - Configurando ambiente...
    echo.
    
    REM Criar ambiente virtual
    echo 🔧 Criando ambiente virtual...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Erro ao criar ambiente virtual
        pause
        exit /b 1
    )
    
    REM Ativar ambiente virtual
    call venv\Scripts\activate.bat
    
    REM Executar setup
    echo 🚀 Executando setup automatizado...
    python setup.py
    if %errorlevel% neq 0 (
        echo ❌ Erro no setup
        pause
        exit /b 1
    )
    
    echo.
    echo ✅ Setup concluído!
    echo.
) else (
    echo 🔄 Ativando ambiente virtual...
    call venv\Scripts\activate.bat
)

REM Menu principal
:menu
echo.
echo 🎯 O que deseja fazer?
echo.
echo 1. 🚀 Executar interface web (Streamlit)
echo 2. 📊 Executar exemplo em linha de comando
echo 3. 🧪 Testar sistema
echo 4. 📁 Abrir pasta de imagens
echo 5. ⚙️ Configurar API do Gemini
echo 6. 🔄 Reinstalar dependências
echo 7. 🚪 Sair
echo.
set /p choice="Digite sua escolha (1-7): "

if "%choice%"=="1" goto run_streamlit
if "%choice%"=="2" goto run_example
if "%choice%"=="3" goto test_system
if "%choice%"=="4" goto open_images
if "%choice%"=="5" goto config_gemini
if "%choice%"=="6" goto reinstall
if "%choice%"=="7" goto exit

echo ❌ Opção inválida!
goto menu

:run_streamlit
echo.
echo 🚀 Iniciando interface web...
echo    Acesse: http://localhost:8501
echo    Pressione Ctrl+C para parar
echo.
streamlit run app.py
goto menu

:run_example
echo.
echo 📊 Executando exemplo...
python exemplo_uso.py
echo.
pause
goto menu

:test_system
echo.
echo 🧪 Testando sistema...
python -c "from rag_system import RAGMultimodalSystem; print('✅ Sistema OK!' if RAGMultimodalSystem().test_system()['overall_success'] else '❌ Sistema com problemas')"
echo.
pause
goto menu

:open_images
echo.
echo 📁 Abrindo pasta de imagens...
if not exist "images" mkdir images
explorer images
echo    Coloque suas imagens na pasta que foi aberta
echo    Formatos suportados: JPG, PNG, BMP, TIFF, WEBP
echo.
pause
goto menu

:config_gemini
echo.
echo ⚙️ Configuração do Google Gemini
echo.
echo Para usar respostas reais do LLM:
echo 1. Obtenha uma API key em: https://makersuite.google.com/app/apikey
echo 2. Edite o arquivo .env
echo 3. Adicione: GEMINI_API_KEY=sua_chave_aqui
echo.
if exist ".env" (
    echo 📄 Abrindo arquivo .env...
    notepad .env
) else (
    echo 📄 Criando arquivo .env...
    echo GEMINI_API_KEY= > .env
    notepad .env
)
echo.
pause
goto menu

:reinstall
echo.
echo 🔄 Reinstalando dependências...
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
echo.
echo ✅ Reinstalação concluída!
pause
goto menu

:exit
echo.
echo 👋 Obrigado por usar o Sistema RAG Multimodal!
echo.
pause
exit /b 0 