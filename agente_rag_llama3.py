import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. Configurações Iniciais
# ==========================================
# Pré-requisitos:
# - Ter o Ollama instalado (https://ollama.com/)
# - Ter o modelo Llama 3 baixado. Para isso, rode no terminal:
#   ollama run llama3

MODEL_NAME = "llama3"
MODEL_NAME = "llama3.2:1b"
# Estamos usando o próprio llama3 para gerar os embeddings (representações vetoriais dos textos)
# para facilitar o setup. Em um ambiente de produção, modelos como 'nomic-embed-text' 
# ou modelos do HuggingFace são mais eficientes para isso.
EMBEDDING_MODEL = "llama3" 
EMBEDDING_MODEL = "llama3.2:1b" 
EMBEDDING_MODEL = "nomic-embed-text" 

# ==========================================
# 2. Carregar e Processar Documentos
# ==========================================
def load_and_split_data(file_path):
    print("Carregando documentos...")
    # Carrega o arquivo de texto
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    
    # Divide o texto em "pedaços" (chunks) menores. 
    # Isso é essencial para que a busca encontre a parte exata da informação
    # e não estoure o limite de tokens do LLM.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # Tamanho máximo de cada pedaço
        chunk_overlap=50    # Sobreposição entre os pedaços para não cortar ideias pela metade
    )
    splits = text_splitter.split_documents(docs)
    return splits

# ==========================================
# 3. Configurar o Banco de Dados Vetorial (Vector Store)
# ==========================================
def setup_vectorstore(splits):
    print("Criando banco de dados vetorial...")
    # O ChromaDB é um banco de dados local que armazena os embeddings.
    # Ele permite buscar semanticamente qual pedaço de texto mais se parece com a pergunta do usuário.
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL)
    )
    # Transforma o banco de dados em uma ferramenta de busca (retriever)
    # k=3 significa que ele vai retornar os 3 pedaços de texto mais relevantes.
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 4. Criar a Cadeia RAG (Search + LLM)
# ==========================================
def create_rag_chain(retriever):
    print("Configurando o Agente Llama 3...")
    
    # Instancia o LLM local rodando via Ollama
    # temperature=0.3 deixa o modelo mais focado e menos "criativo" (ideal para RAG)
    llm = ChatOllama(model=MODEL_NAME, temperature=0.3)
    
    # Prompt do Sistema Genérico para RAG
    # Define o comportamento do agente e a regra de ouro do RAG: usar apenas o contexto.
    system_prompt = """Você é um assistente prestativo, educado e inteligente.
    Use **apenas** os seguintes pedaços de contexto recuperado para responder à pergunta do usuário.
    Se a resposta não estiver no contexto, diga claramente que você não tem essa informação, não tente adivinhar.
    Mantenha a resposta clara e concisa.

    Contexto recuperado:
    {context}
    
    Pergunta do Usuário: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # Função auxiliar para juntar os pedaços de texto encontrados em uma única string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Constrói o Pipeline (Chain) do RAG usando a sintaxe moderna (LCEL) do LangChain:
    # 1. Pega a pergunta e busca o contexto no retriever
    # 2. Formata tudo no Prompt
    # 3. Envia para o LLM
    # 4. Extrai a string final da resposta
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ==========================================
# 5. Execução do Script
# ==========================================
if __name__ == "__main__":
    # 5.1. Criar um arquivo de dados de exemplo (se não existir)
    sample_file = "dados_exemplo.txt"
    if not os.path.exists(sample_file):
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("A Antigravity é uma IA avançada desenvolvida por especialistas para auxiliar programadores no dia a dia.\n")
            f.write("O projeto 'Agente Inteligente' foi criado em 2026 com o objetivo de revolucionar a automação local nas empresas.\n")
            f.write("A linguagem de programação oficial do projeto é Python devido ao seu vasto ecossistema de bibliotecas de Inteligência Artificial.\n")
            f.write("A sede do projeto fica localizada na cidade de São Paulo, mas os contribuidores são de todo o Brasil.\n")
    
    # 5.2. Montar o sistema
    splits = load_and_split_data(sample_file)
    retriever = setup_vectorstore(splits)
    agent_chain = create_rag_chain(retriever)
    
    print("\n" + "="*50)
    print("🤖 Agente Llama 3 (RAG) Iniciado!")
    print("Agente Llama 3 (RAG) Iniciado!")
    print("Base de dados 'dados_exemplo.txt' carregada com sucesso.")
    print("Faça uma pergunta sobre o conteúdo do arquivo (ou digite 'sair').")
    print("="*50 + "\n")
    
    # 5.3. Loop de interação com o usuário
    while True:
        try:
            user_input = input("Você: ")
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando agente. Até logo!")
                break
            
            if not user_input.strip():
                continue
                
            print("Agente: Lendo base de dados e pensando...")
            # Invoca a cadeia (Chain) passando a pergunta
            response = agent_chain.invoke(user_input)
            print(f"🤖 Agente: {response}\n")
            print(f"Agente: {response}\n")

            
        except KeyboardInterrupt:
            print("\nEncerrando agente. Até logo!")
            break
        except Exception as e:
            print(f"\nOcorreu um erro: {e}\n")

