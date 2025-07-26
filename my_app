import streamlit as st
import pandas as pd
import joblib # Use joblib para carregar ficheiros .pkl, é mais robusto
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Configuração da Página e Carregamento de Dados ---
st.set_page_config(
    page_title="Dashboard de Risco de Crédito",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções de Carregamento em Cache ---
@st.cache_resource
def load_model(caminho_modelo):
    """Carrega o pipeline treinado a partir de um ficheiro .pkl."""
    try:
        # Usamos joblib.load() que é o padrão para carregar modelos scikit-learn
        return joblib.load(caminho_modelo)
    except FileNotFoundError:
        return None

@st.cache_data
def load_data(caminho_dados):
    """Carrega os dados para a análise exploratória."""
    try:
        return pd.read_csv(caminho_dados)
    except FileNotFoundError:
        return None

# --- Barra Lateral e Carregamento dos Ficheiros ---
st.sidebar.title("🏦 Dashboard de Risco")
st.sidebar.markdown("---")
st.sidebar.header("Configuração de Ficheiros")

# Peça ao utilizador para fornecer os nomes dos ficheiros
caminho_modelo_pkl = st.sidebar.text_input("Nome do seu ficheiro de modelo:", "best.pkl")
caminho_dados_csv = st.sidebar.text_input("Nome do seu ficheiro de dados:", "dados1.csv")

pipeline = load_model(caminho_modelo_pkl)
dados = load_data(caminho_dados_csv)

st.sidebar.markdown("---")

# --- Navegação ---
if dados is not None:
    st.sidebar.header("Filtros da Análise")
    selected_finalidade = st.sidebar.multiselect(
        "Filtrar por Finalidade",
        options=dados['Finalidade'].unique(),
        default=dados['Finalidade'].unique()
    )
    dados_filtrados = dados[dados['Finalidade'].isin(selected_finalidade)]
else:
    dados_filtrados = pd.DataFrame()

pagina = st.sidebar.radio(
    "Navegue pelas seções:",
    ["📊 Dashboard Geral", "📈 Análise Exploratória", "🧠 Detalhes do Modelo", "⚙️ Simulador de Risco", "💼 Impacto no Negócio"]
)
st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido como uma ferramenta de suporte à decisão para análise de crédito.")


# --- Verificação de Erro ---
if pipeline is None or dados is None:
    st.error(f"ERRO: Não foi possível carregar os ficheiros. Verifique se os nomes '{caminho_modelo_pkl}' e '{caminho_dados_csv}' estão corretos e se os ficheiros estão na mesma pasta que o script.")
    st.stop()


# --- Conteúdo das Páginas ---

# PÁGINA 1: DASHBOARD GERAL
if pagina == "📊 Dashboard Geral":
    st.title("📊 Dashboard Geral do Modelo de Risco")
    st.markdown("Visão geral do desempenho do modelo e da distribuição dos dados.")

    # Métricas Chave
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall do Modelo", "66.8%", "Foco Principal", help="Capacidade de identificar os 'maus pagadores'.")
    col2.metric("Precisão do Modelo", "96.9%", "Alta Confiabilidade", help="Assertividade do modelo ao classificar um cliente como 'mau pagador'.")
    col3.metric("Clientes no Dataset", f"{dados_filtrados.shape[0]}", "Total Analisado")
    col4.metric("Taxa de Inadimplência", f"{(dados_filtrados['Cliente'] == 'mau pagador').mean():.2%}", "Observada")

    st.markdown("---")

    # Gráficos do Dashboard
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Importância das Variáveis (Features)")
        try:
            # Tenta extrair a importância das features do pipeline
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            importances = pipeline.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
            
            fig_imp = px.bar(feature_importance_df, x='importance', y='feature', orientation='h', title='Top 10 Variáveis Mais Influentes')
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível extrair a importância das features do seu modelo. O pipeline pode ter uma estrutura diferente. Erro: {e}")

    with col_b:
        st.subheader("Distribuição de Clientes")
        cliente_counts = dados_filtrados['Cliente'].value_counts()
        fig_pie = px.pie(values=cliente_counts.values, names=cliente_counts.index, title='Proporção de Bons vs. Maus Pagadores', hole=.3)
        st.plotly_chart(fig_pie, use_container_width=True)

# PÁGINA 2: ANÁLISE EXPLORATÓRIA
elif pagina == "📈 Análise Exploratória":
    st.title("📈 Análise Exploratória Interativa")
    st.markdown("Explore as relações entre as variáveis do conjunto de dados filtrado.")

    tab1, tab2, tab3 = st.tabs(["Análise Univariada", "Análise Bivariada", "Análise Categórica"])

    with tab1:
        st.subheader("Análise de uma única variável numérica")
        num_col_select = st.selectbox("Selecione uma variável numérica:", options=dados.select_dtypes(include=np.number).columns)
        fig_hist = px.histogram(dados_filtrados, x=num_col_select, color='Cliente', marginal='box',
                                title=f'Distribuição de {num_col_select} por tipo de cliente')
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader("Relação entre duas variáveis numéricas")
        col_x = st.selectbox("Selecione a variável para o eixo X:", options=dados.select_dtypes(include=np.number).columns, index=0)
        col_y = st.selectbox("Selecione a variável para o eixo Y:", options=dados.select_dtypes(include=np.number).columns, index=1)
        fig_scatter = px.scatter(dados_filtrados, x=col_x, y=col_y, color='Cliente',
                                 title=f'{col_y} vs. {col_x}', hover_data=['Finalidade'])
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.subheader("Análise de variáveis categóricas")
        cat_col_select = st.selectbox("Selecione uma variável categórica:", options=dados.select_dtypes(include='object').drop('Cliente', axis=1).columns)
        fig_bar = px.bar(dados_filtrados.groupby([cat_col_select, 'Cliente']).size().reset_index(name='count'),
                         x=cat_col_select, y='count', color='Cliente', barmode='group',
                         title=f'Contagem por {cat_col_select} e tipo de cliente')
        st.plotly_chart(fig_bar, use_container_width=True)

# PÁGINA 3: DETALHES DO MODELO
elif pagina == "🧠 Detalhes do Modelo":
    st.title("🧠 Análise Profunda do Modelo")
    st.markdown("Aqui exploramos o comportamento e a performance do modelo carregado.")
    
    tab_matriz, tab_curvas = st.tabs(["Matriz de Confusão", "Curvas de Performance"])

    with tab_matriz:
        st.subheader("Matriz de Confusão")
        st.image("https://i.imgur.com/zW2q2oX.png", caption="Matriz de Confusão de exemplo do Modelo Otimizado.", use_column_width=True)
        st.markdown("""
        **Nota:** Esta é uma imagem de exemplo. A performance real do seu modelo pode ser diferente.
        - **Canto inferior esquerdo (Falso Negativo):** O erro mais caro. Maus pagadores que o modelo deixou passar. **O objetivo é minimizar este número.**
        """)

    with tab_curvas:
        st.subheader("Curva de Precisão vs. Recall (PR Curve)")
        st.info("Esta curva ajuda a visualizar o trade-off entre Precisão e Recall. Idealmente, queremos estar no canto superior direito.")
        # Simulação de uma curva PR para fins visuais
        recall_vals = np.linspace(0.6, 1.0, 100)
        precision_vals = 0.97 - 0.4 * (recall_vals - 0.6) + np.random.normal(0, 0.02, 100)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name='Curva PR'))
        fig_pr.add_trace(go.Scatter(x=[0.668], y=[0.969], mode='markers', marker=dict(color='red', size=12), name='Exemplo de Ponto Operacional'))
        fig_pr.update_layout(title='Curva de Precisão vs. Recall (Exemplo)', xaxis_title='Recall', yaxis_title='Precisão')
        st.plotly_chart(fig_pr, use_container_width=True)

# PÁGINA 4: SIMULADOR DE RISCO
elif pagina == "⚙️ Simulador de Risco":
    st.title("⚙️ Simulador Interativo de Risco de Crédito")
    st.markdown("Insira os dados de um novo solicitante para obter uma análise de risco em tempo real.")

    with st.expander("Clique aqui para preencher o formulário do cliente", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            finalidade = st.selectbox("Finalidade", options=dados['Finalidade'].unique())
            emprestimo_valor = st.slider("Valor do Empréstimo (R$)", 1000, 200000, 50000)
            valor_do_bem = st.slider("Valor do Bem (R$)", 1000, 1000000, 100000)
        with col2:
            emprego = st.selectbox("Tipo de Emprego", options=dados['Emprego'].unique())
            tempo_emprego = st.slider("Tempo no Emprego (anos)", 0.0, 40.0, 5.0)
            tempo_cliente = st.slider("Tempo como Cliente (meses)", 1.0, 500.0, 120.0)
        with col3:
            negativos = st.slider("Registros Negativos", 0, 10, 0)
            atrasos = st.slider("Número de Atrasos", 0, 20, 0)
            rds = st.slider("Renda Comprometida (RDS %)", 0.0, 100.0, 30.0)
            lc_recente = st.slider("Linhas de Crédito Recentes", 0, 10, 1)
            lc_atual = st.slider("Linhas de Crédito Atuais", 0, 20, 5)

    if st.button("Analisar Risco do Cliente", type="primary"):
        input_data = pd.DataFrame({
            'Empréstimo': [emprestimo_valor], 'ValorDoBem': [valor_do_bem], 'Finalidade': [finalidade],
            'Emprego': [emprego], 'TempoEmprego': [tempo_emprego], 'Negativos': [negativos],
            'Atrasos': [atrasos], 'TempoCliente': [tempo_cliente], 'LC-Recente': [lc_recente],
            'LC-Atual': [lc_atual], 'RDS': [rds]
        })

        prediction_proba = pipeline.predict_proba(input_data)[0]
        prob_mau_pagador = prediction_proba[1] # Assumindo que a classe 1 é 'mau pagador'

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob_mau_pagador * 100,
            title = {'text': "Probabilidade de Inadimplência (%)"},
            gauge = {'axis': {'range': [None, 100]},
                     'steps' : [{'range': [0, 30], 'color': "green"}, {'range': [30, 60], 'color': "orange"}, {'range': [60, 100], 'color': "red"}],
                     'bar': {'color': "black"}}))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if prob_mau_pagador > 0.5:
             st.error(f"**ALERTA DE RISCO:** O modelo classificou este cliente como **potencial Mau Pagador** com {prob_mau_pagador:.2%} de probabilidade.")
        else:
             st.success(f"**BAIXO RISCO:** O modelo classificou este cliente como **potencial Bom Pagador**. A probabilidade de inadimplência é de apenas {prob_mau_pagador:.2%}.")

# PÁGINA 5: IMPACTO NO NEGÓCIO
elif pagina == "💼 Impacto no Negócio":
    st.title("💼 Calculadora de Impacto Financeiro")
    st.markdown("Estime o valor financeiro que o modelo pode economizar para a empresa.")

    st.subheader("Parâmetros da Simulação")
    col1, col2, col3 = st.columns(3)
    with col1:
        valor_medio_emprestimo = st.number_input("Valor Médio por Empréstimo (R$)", min_value=1000, value=25000, step=1000)
    with col2:
        taxa_perda = st.slider("Taxa de Perda sobre Inadimplência (%)", 10, 100, 60)
    with col3:
        fn_evitados = st.number_input("Falsos Negativos Evitados pelo Modelo", min_value=0, value=9, step=1)

    prejuizo_por_fn = valor_medio_emprestimo * (taxa_perda / 100)
    economia_total = fn_evitados * prejuizo_por_fn

    st.markdown("---")
    st.subheader("Resultados da Simulação")
    st.info(f"Considerando os **{fn_evitados} Falsos Negativos evitados** pelo modelo:")
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Prejuízo Evitado por Cliente", f"R$ {prejuizo_por_fn:,.2f}")
    col_res2.metric("Economia Total Gerada pelo Modelo", f"R$ {economia_total:,.2f}", "Valor Agregado")
