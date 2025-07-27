import streamlit as st
import pandas as pd
import joblib # Use joblib para carregar ficheiros .pkl, é mais robusto
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Risco de Crédito",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções de Carregamento e Pré-processamento ---
@st.cache_resource
def load_model(caminho_modelo):
    """Carrega o modelo treinado a partir de um ficheiro .pkl."""
    try:
        return joblib.load(caminho_modelo)
    except Exception as e:
        st.error(f"ERRO AO CARREGAR O MODELO: Verifique o nome do ficheiro e as dependências. Erro: {e}")
        return None

@st.cache_data
def load_data(caminho_dados):
    """Carrega os dados para a análise exploratória."""
    try:
        return pd.read_csv(caminho_dados)
    except Exception as e:
        st.error(f"ERRO AO CARREGAR OS DADOS: Verifique o nome do ficheiro '{caminho_dados}'. Erro: {e}")
        return None

def preprocess_for_prediction(df_to_predict, reference_df):
    """
    Prepara um DataFrame para previsão, aplicando One-Hot Encoding
    e alinhando as colunas com base num DataFrame de referência.
    Esta função é um workaround para quando o .pkl não contém o pipeline de pré-processamento.
    """
    # Identifica colunas categóricas do DataFrame de referência
    categorical_features = reference_df.select_dtypes(include=['object']).columns
    if 'Cliente' in categorical_features:
        categorical_features = categorical_features.drop('Cliente')

    # Aplica get_dummies ao DataFrame de entrada
    df_processed = pd.get_dummies(df_to_predict, columns=categorical_features)

    # Cria colunas dummy a partir do DataFrame de referência para obter o conjunto completo de colunas
    reference_processed = pd.get_dummies(reference_df.drop('Cliente', axis=1), columns=categorical_features)

    # Alinha as colunas do df de entrada com as do df de referência
    df_aligned = df_processed.reindex(columns=reference_processed.columns, fill_value=0)
    
    return df_aligned

# --- Barra Lateral e Carregamento dos Ficheiros ---
st.sidebar.title("🏦 Dashboard de Risco")
st.sidebar.markdown("---")
st.sidebar.header("Configuração de Ficheiros")

caminho_modelo_pkl = st.sidebar.text_input("Nome do seu ficheiro de modelo:", "best.pkl")
caminho_dados_csv = st.sidebar.text_input("Nome do seu ficheiro de dados:", "dados1.csv")

model = load_model(caminho_modelo_pkl)
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
if model is None or dados is None:
    st.stop()


# --- Conteúdo das Páginas ---

# PÁGINA 1: DASHBOARD GERAL
if pagina == "📊 Dashboard Geral":
    st.title("📊 Dashboard Geral do Modelo de Risco")
    st.markdown("Visão geral do desempenho do modelo e da distribuição dos dados.")

    # --- A CORREÇÃO ESTÁ AQUI ---
    # As métricas agora são dinâmicas e baseadas nos cálculos acima
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall do Modelo", f"{recall:.2%}", help="Capacidade de identificar os 'maus pagadores'.")
    col2.metric("Precisão do Modelo", f"{precision:.2%}", help="Assertividade do modelo ao classificar um cliente como 'mau pagador'.")
    col3.metric("Acurácia Geral", f"{accuracy:.2%}", help="Percentual geral de acertos do modelo.")
    col4.metric("Taxa de Inadimplência", f"{(dados_filtrados['Cliente'] == 'mau pagador').mean():.2%}", "Observada nos dados filtrados")
    # --- FIM DA CORREÇÃO ---

    st.markdown("---")

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
        fig_scatter = px.scatter(dados_filtrados, x=col_x, y=col_y, color='Cliente', title=f'{col_y} vs. {col_x}', hover_data=['Finalidade'], render_mode='svg')
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

    try:
        X_raw = dados.drop('Cliente', axis=1)
        y_true = dados['Cliente'].map({'bom pagador': 0, 'mau pagador': 1})
        X_processed = preprocess_for_prediction(X_raw, dados)
        y_pred = model.predict(X_processed)
        y_proba = model.predict_proba(X_processed)[:, 1]
        cm = confusion_matrix(y_true, y_pred)
        precision_points, recall_points, _ = precision_recall_curve(y_true, y_proba)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
    except Exception as e:
        st.warning(f"Não foi possível gerar os gráficos de performance do modelo. Erro: {e}")
        cm = np.array([[0, 0], [0, 0]])
        precision_points, recall_points = [0], [0]
        recall, precision = 0.0, 0.0

    with tab_matriz:
        st.subheader("Matriz de Confusão Dinâmica")
        z = cm
        x = ['Bom Pagador (Previsto)', 'Mau Pagador (Previsto)']
        y = ['Bom Pagador (Real)', 'Mau Pagador (Real)']
        z_text = [[str(y) for y in x] for x in z]
        fig_cm = go.Figure(data=go.Heatmap(
                   z=z, x=x, y=y, hoverongaps=False, text=z_text,
                   texttemplate="%{text}", colorscale='Greens'))
        fig_cm.update_layout(title_text='<i><b>Matriz de Confusão do Modelo Carregado</b></i>')
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("Esta matriz é gerada **dinamicamente** com base no seu modelo e dados.")

    with tab_curvas:
        st.subheader("Curva de Precisão vs. Recall (PR Curve)")
        st.info("Esta curva ajuda a visualizar o trade-off entre Precisão e Recall.")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall_points, y=precision_points, mode='lines', name='Curva PR do Modelo'))
        fig_pr.add_trace(go.Scatter(x=[recall], y=[precision], mode='markers', marker=dict(color='red', size=12), name='Ponto Operacional Atual'))
        fig_pr.update_layout(title='Curva de Precisão vs. Recall Dinâmica', xaxis_title='Recall', yaxis_title='Precisão')
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
        # Criar DataFrame com os dados do formulário
        input_data_dict = {
            'Empréstimo': [emprestimo_valor], 'ValorDoBem': [valor_do_bem], 'Finalidade': [finalidade],
            'Emprego': [emprego], 'TempoEmprego': [tempo_emprego], 'Negativos': [negativos],
            'Atrasos': [atrasos], 'TempoCliente': [tempo_cliente], 'LC-Recente': [lc_recente],
            'LC-Atual': [lc_atual], 'RDS': [rds]
        }
        input_data = pd.DataFrame(input_data_dict)
        
        # --- A CORREÇÃO ESTÁ AQUI ---
        # Obter a ordem das colunas diretamente do modelo treinado para garantir a correspondência.
        try:
            # A forma mais robusta é usar o atributo feature_names_in_ do pipeline
            X_train_columns = pipeline.feature_names_in_
            input_data = input_data[X_train_columns]
        except AttributeError:
            # Fallback para o caso de o modelo não ser um pipeline sklearn
            st.warning("Não foi possível obter a ordem das features do modelo. Usando a ordem do CSV. Isso pode causar erros se não for idêntica.")
            X_train_columns = dados.drop('Cliente', axis=1).columns
            input_data = input_data[X_train_columns]
        # --- FIM DA CORREÇÃO ---


        prediction_proba = pipeline.predict_proba(input_data)[0]
        prob_mau_pagador = prediction_proba[1] 

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
