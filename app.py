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

# Configurando pagina web
st.set_page_config(
    page_title="Estoque",
    page_icon="📸",
    layout="wide"
)
openai.api_key = st.secrets["OPENAI_API_KEY"]
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Escreva uma breve descrição de um erro.",
    max_tokens=50
)
print(response)

# Cabeçalho da página
st.header("Assistente de Estoque")

# listar modelo e opções de LLM em um menu
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

st.sidebar.markdown(
    "### Sobre o Sistema"
)
st.sidebar.markdown(
    "Este agente utiliza da inteligência artificial para consultar um banco de dados de estoque."
)
# Conversando com o usuário
st.write("Faça perguntas sobre o estoque de produtos, preços e reposições.")
# Input do usuário
UserQuestion = st.text_input("O que deseja saber sobre o estoque?")

# Modelo que o usuário quer usar
model = ChatOpenAI(
    model=selected_model,
)

# Conectando ao banco de dados
data = SQLDatabase.from_uri("sqlite:///estoque.db")

# Criando toolkit do database SQL
toolkit = SQLDatabaseToolkit(
    db=data,
    llm=model,
)
system_message = hub.pull("hwchase17/react")

# Criando o agente
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

# Criando prompt para perguntar
prompt = """
Use as ferramentas mecessárias para responder perguntas relacionadas ao estoque de produtos.
Você fornecerá insights sobre produtos, preços, reposição de estoque e relatórios conforme solicitado
pelo usuário. A resposta final deve ter uma formatação amigável de visualização para o usuário.
Sempre responsa em português brasileiro.
Pergunta: {pergunta}
"""

prompt_template = PromptTemplate.from_template(prompt)

# criando botão de consultar
if st.button("Consultar"):
    if UserQuestion:
        # Icone de carregamento
        with st.spinner("Consultando o banco de dados..."):
            formatted_prompt = prompt_template.format(pergunta=UserQuestion)
            output = agent_executor.invoke(
                {"input": formatted_prompt}
            )
        # Renderizando texto ao usuário
        st.markdown(output.get("output"))
    else:
        st.warning("Por favor, faça uma pergunta")
