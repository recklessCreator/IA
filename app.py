import os
import streamlit as st
from decouple import config
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

# Configurando página web
st.set_page_config(
    page_title="Estoque",
    page_icon="📸",
)

# Cabeçalho da página
st.header("Assistente de Estoque")

# Listar modelo e opções de LLM em um menu
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

# Criando toolkit do banco de dados SQL
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
    handle_parsing_errors=True  # Permitir que o agente tente novamente em caso de erro de parsing
)

# Criando o prompt para perguntar
prompt = """
Use as ferramentas necessárias para responder perguntas relacionadas ao estoque de produtos.
Você fornecerá insights sobre produtos, preços, reposição de estoque e relatórios conforme solicitado
pelo usuário. A resposta final deve ser concisa, clara e amigável, e formatada como um resumo direto.
Sempre responda em português brasileiro.
Pergunta: {pergunta}
"""

prompt_template = PromptTemplate.from_template(prompt)

# Criando botão de consultar
if st.button("Consultar"):
    if UserQuestion:
        # Icone de carregamento
        with st.spinner("Consultando o banco de dados..."):
            formatted_prompt = prompt_template.format(pergunta=UserQuestion)
            response = agent_executor.invoke({"input": formatted_prompt})
            
            # Exibindo a resposta completa para inspeção (debugging)
            st.write("Estrutura da resposta:", response)

            # Tentando acessar a resposta no formato correto
            if 'text' in response:
                st.markdown(response['text'])
            elif 'output' in response:
                # Caso a resposta esteja em 'output'
                st.markdown(response['output'])
            else:
                # Caso a chave correta não seja encontrada, mostrando a resposta inteira
                st.warning("Resposta não encontrada no formato esperado.")
    else:
        st.warning("Por favor, faça uma pergunta.")
