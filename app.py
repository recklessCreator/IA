import os
import openai
import streamlit as st
from dotenv import load_dotenv
from decouple import config
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

# Configurando a página do Streamlit
st.set_page_config(
    page_title="Assistente de Estoque",
    page_icon="📦",
    layout="wide",
)

# Configurando a chave da API OpenAI
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Chave da API OpenAI não encontrada no arquivo de configurações. Verifique 'Secrets' no Streamlit Cloud.")

# Cabeçalho da página
st.header("Assistente de Estoque")

# Listar modelos de LLM disponíveis
model_options = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o-mini",
    "gpt-4o",
]

selected_model = st.sidebar.selectbox(
    label="Selecione o modelo de LLM",
    options=model_options,
)

# Informações adicionais na barra lateral
st.sidebar.markdown("### Sobre o Sistema")
st.sidebar.markdown(
    "Este agente utiliza inteligência artificial para consultar um banco de dados de estoque."
)

# Entrada do usuário
UserQuestion = st.text_input("O que deseja saber sobre o estoque?")

# Configurando o modelo selecionado
model = ChatOpenAI(
    model=selected_model,
)

# Conectando ao banco de dados SQLite
try:
    data = SQLDatabase.from_uri("sqlite:///estoque.db")
except Exception as e:
    st.error(f"Erro ao conectar ao banco de dados: {e}")

# Criando o toolkit SQL
toolkit = SQLDatabaseToolkit(
    db=data,
    llm=model,
)

# Sistema de mensagens do agente
try:
    system_message = hub.pull("hwchase17/react")
except Exception as e:
    st.error(f"Erro ao carregar sistema de mensagens: {e}")

# Criando o agente
try:
    agent = create_react_agent(
        llm=model,
        tools=toolkit.get_tools(),
        prompt=system_message,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolkit.get_tools(),
        verbose=True,
    )
except Exception as e:
    st.error(f"Erro ao configurar o agente: {e}")

# Template do prompt
prompt = """
Use as ferramentas necessárias para responder perguntas relacionadas ao estoque de produtos.
Você fornecerá insights sobre produtos, preços, reposição de estoque e relatórios conforme solicitado
pelo usuário. A resposta final deve ter uma formatação amigável de visualização para o usuário.
Sempre responda em português brasileiro.
Pergunta: {pergunta}
"""

prompt_template = PromptTemplate.from_template(prompt)

# Botão para consultar
if st.button("Consultar"):
    if UserQuestion:
        # Ícone de carregamento
        with st.spinner("Consultando o banco de dados..."):
            try:
                formatted_prompt = prompt_template.format(pergunta=UserQuestion)
                output = agent_executor.invoke(
                    {"input": formatted_prompt}
                )
                # Exibir resultado
                st.markdown(output.get("output"))
            except Exception as e:
                st.error(f"Erro ao executar a consulta: {e}")
    else:
        st.warning("Por favor, insira uma pergunta.")
