import os
import streamlit as st
from decouple import config
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

# Configurar a chave da API do OpenAI
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

# Configurando página do Streamlit
st.set_page_config(
    page_title="Estoque",
    page_icon="📸",
)

# Cabeçalho da página
st.header("Assistente de Estoque")

# Menu de seleção de modelo
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

# Sobre o sistema
st.sidebar.markdown("### Sobre o Sistema")
st.sidebar.markdown(
    "Este agente utiliza da inteligência artificial para consultar um banco de dados de estoque."
)

# Pergunta do usuário
st.write("Faça perguntas sobre o estoque de produtos, preços e reposições.")
UserQuestion = st.text_input("O que deseja saber sobre o estoque?")

# Modelo selecionado pelo usuário
model = ChatOpenAI(
    model=selected_model,
)

# Conectar ao banco de dados SQLite
data = SQLDatabase.from_uri("sqlite:///estoque.db")

# Criar toolkit para consultar o banco de dados
toolkit = SQLDatabaseToolkit(
    db=data,
    llm=model,
)

# Carregar o modelo do sistema
system_message = hub.pull("hwchase17/react")

# Criar o agente de consulta
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

# Criando o prompt
prompt = """
Use as ferramentas necessárias para responder perguntas relacionadas ao estoque de produtos.
Você fornecerá insights sobre produtos, preços, reposição de estoque e relatórios conforme solicitado
pelo usuário. A resposta final deve ter uma formatação amigável de visualização para o usuário.
Sempre responda em português brasileiro.
Pergunta: {pergunta}
"""

prompt_template = PromptTemplate.from_template(prompt)

# Botão de consulta
if st.button("Consultar"):
    if UserQuestion:
        # Ícone de carregamento
        with st.spinner("Consultando o banco de dados..."):
            try:
                formatted_prompt = prompt_template.format(pergunta=UserQuestion)
                # Invocar o agente
                response = agent_executor.invoke({"input": formatted_prompt})
                # Renderizar resposta
                st.markdown(response['text'])  # Ajuste conforme a estrutura da resposta
            except Exception as e:
                st.error(f"Erro ao processar a consulta: {e}")
    else:
        st.warning("Por favor, faça uma pergunta.")
