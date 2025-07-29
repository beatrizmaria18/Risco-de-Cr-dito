# Arquivo Único: app_final.py
# ------------------------------------
# OBJETIVO: Versão final com pré-processamento manual para garantir compatibilidade com o modelo.
#
# Para rodá-lo, execute no seu terminal:
# streamlit run app_final.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    recall_score, precision_score, accuracy_score,
    confusion_matrix, precision_recall_curve, classification_report
)

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Risco de Crédito",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

OPTIMAL_THRESHOLD = 0.42

# --- Funções de Carregamento Melhoradas ---
@st.cache_resource
def load_model(model_path):
    """Carrega o modelo corretamente, tratando casos onde foi salvo como dicionário"""
    try:
        model = joblib.load(model_path)
        
        # Se o modelo foi salvo como dicionário (contendo pipeline e metadados)
        if isinstance(model, dict):
            if 'model' in model:
                return model['model']
            elif 'pipeline' in model:
                return model['pipeline']
            else:
                st.error("Arquivo de modelo inválido: dicionário sem chave 'model' ou 'pipeline'")
                return None
        return model
    except Exception as e:
        st.error(f"ERRO AO CARREGAR MODELO: {str(e)}")
        return None

@st.cache_data
def load_data(data_path):
    """Carrega e pré-processa os dados"""
    try:
        dados = pd.read_csv(data_path)
        dados['RDS'] = dados['RDS'].astype(str).str.replace('%', '').astype(float)
        return dados
    except Exception as e:
        st.error(f"ERRO AO CARREGAR DADOS: {str(e)}")
        return None

# --- Carregamento dos Arquivos ---
st.sidebar.title("🏦 Dashboard de Risco")
st.sidebar.markdown("---")
st.sidebar.header("Configuração de Arquivos")

model_path = st.sidebar.text_input("Caminho do modelo:", "best.pkl")
data_path = st.sidebar.text_input("Caminho dos dados:", "dados1.csv")

model = load_model(model_path)
dados = load_data(data_path)

# --- Verificação de Carregamento ---
if model is None or dados is None:
    st.error("Não foi possível carregar os arquivos necessários. Verifique os caminhos.")
    st.stop()




# --- Função para preparar os dados para previsão ---
def prepare_features(df):
    """Cria as features adicionais necessárias para o modelo"""
    df = df.copy()
    
    # Feature engineering (idêntico ao que foi feito no treinamento)
    df['Risco_Atrasos'] = df['Atrasos'] * df['Negativos']
    df['Historico_Risco'] = df['TempoCliente'] / (df['Atrasos'] + 1e-6)  # +1e-6 para evitar divisão por zero
    df['Alavancagem'] = df['Empréstimo'] / (df['ValorDoBem'] + 0.001)  # +0.001 para evitar divisão por zero
    
    return df






# --- Navegação ---
pages = {
    "📊 Dashboard Geral": "dashboard",
    "📈 Análise Exploratória": "analise",
    "🧠 Detalhes do Modelo": "modelo",
    "⚙️ Simulador de Risco": "simulador",
    "💼 Impacto no Negócio": "impacto"
}

selected_page = st.sidebar.radio("Navegue pelas seções:", list(pages.keys()))
st.sidebar.markdown("---")

# Filtros da Análise (apenas para páginas que usam dados)
if selected_page in ["📊 Dashboard Geral", "📈 Análise Exploratória"]:
    st.sidebar.header("Filtros da Análise")
    selected_finalidade = st.sidebar.multiselect(
        "Filtrar por Finalidade",
        options=dados['Finalidade'].unique(),
        default=dados['Finalidade'].unique()
    )
    dados_filtrados = dados[dados['Finalidade'].isin(selected_finalidade)]
else:
    dados_filtrados = pd.DataFrame()

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido como uma ferramenta de suporte à decisão para análise de crédito.")

# --- Conteúdo das Páginas ---

# PÁGINA 1: DASHBOARD GERAL
if selected_page == "📊 Dashboard Geral":
    st.title("📊 Dashboard do Modelo de Risco")
    
    try:
        if selected_page == "📊 Dashboard Geral":
    st.title("📊 Dashboard do Modelo de Risco")
    
    try:
        # Preparar dados para avaliação
        X = dados.drop('Cliente', axis=1)
        X = prepare_features(X)  # Adicionar esta linha
        
        y_true = dados['Cliente'].map({'bom pagador': 0, 'mau pagador': 1})
        
        # Verificar se o modelo tem predict_proba
        if not hasattr(model, 'predict_proba'):
            st.error("O modelo carregado não suporta previsões de probabilidade")
            st.stop()
            
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
        
        # Calcular métricas
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Exibir métricas
        cols = st.columns(3)
        cols[0].metric("Recall", f"{recall:.2%}", 
                      help="Capacidade de identificar maus pagadores")
        cols[1].metric("Precisão", f"{precision:.2%}", 
                      help="Acerto ao classificar como mau pagador")
        cols[2].metric("Acurácia", f"{accuracy:.2%}", 
                      help="Percentual total de acertos")
        
        # Gráfico de distribuição
        fig = px.pie(
            dados['Cliente'].value_counts().reset_index(),
            values='count',
            names='Cliente',
            title='Distribuição de Clientes'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro ao calcular métricas: {str(e)}")

# PÁGINA 2: ANÁLISE EXPLORATÓRIA
elif selected_page == "📈 Análise Exploratória":
    st.title("📈 Análise Exploratória Interativa")
    st.markdown("Explore as relações entre as variáveis do conjunto de dados filtrado.")
    
    if not dados_filtrados.empty:
        # Criar features consistentes com o modelo
        dados_filtrados['Risco_Atrasos'] = dados_filtrados['Atrasos'] * dados_filtrados['Negativos']
        dados_filtrados['Historico_Risco'] = dados_filtrados['TempoCliente'] / (dados_filtrados['Atrasos'] + 1)
        dados_filtrados['Alavancagem'] = dados_filtrados['Empréstimo'] / (dados_filtrados['ValorDoBem'] + 0.001)
    
    tab1, tab2, tab3 = st.tabs(["Análise Univariada", "Análise Bivariada", "Análise Categórica"])
    
    with tab1:
        st.subheader("Análise de uma única variável numérica")
        num_cols = dados_filtrados.select_dtypes(include=np.number).columns.tolist()
        num_col_select = st.selectbox("Selecione uma variável numérica:", options=num_cols)
        fig_hist = px.histogram(dados_filtrados, x=num_col_select, color='Cliente', marginal='box', 
                               title=f'Distribuição de {num_col_select} por tipo de cliente')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        st.subheader("Relação entre duas variáveis numéricas")
        num_cols = dados_filtrados.select_dtypes(include=np.number).columns.tolist()
        col_x = st.selectbox("Selecione a variável para o eixo X:", options=num_cols, index=0)
        col_y = st.selectbox("Selecione a variável para o eixo Y:", options=num_cols, index=1)
        fig_scatter = px.scatter(dados_filtrados, x=col_x, y=col_y, color='Cliente', 
                               title=f'{col_y} vs. {col_x}', hover_data=['Finalidade'])
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.subheader("Análise de variáveis categóricas")
        cat_cols = dados_filtrados.select_dtypes(include='object').drop('Cliente', axis=1, errors='ignore').columns
        if len(cat_cols) > 0:
            cat_col_select = st.selectbox("Selecione uma variável categórica:", options=cat_cols)
            fig_bar = px.bar(dados_filtrados.groupby([cat_col_select, 'Cliente']).size().reset_index(name='count'), 
                            x=cat_col_select, y='count', color='Cliente', barmode='group', 
                            title=f'Contagem por {cat_col_select} e tipo de cliente')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Nenhuma variável categórica disponível para análise.")

# PÁGINA 3: DETALHES DO MODELO
elif selected_page == "🧠 Detalhes do Modelo":
    st.title("🧠 Análise Profunda do Modelo")
    st.markdown("Aqui exploramos o comportamento e a performance do modelo carregado.")
    
    try:
        # Preparar os dados com todas as features necessárias
        X_raw = dados.drop('Cliente', axis=1)
        X_raw = prepare_features(X_raw)  # Adiciona as features extras
        
        y_true = dados['Cliente'].map({'bom pagador': 0, 'mau pagador': 1})
        y_proba = model.predict_proba(X_raw)[:, 1]
        y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        precision_points, recall_points, thresholds = precision_recall_curve(y_true, y_proba)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        
    except Exception as e:
        st.error(f"Erro ao gerar gráficos de performance: {str(e)}")
        st.error("Verifique se todas as colunas necessárias estão presentes nos dados.")
        cm = np.array([[0, 0], [0, 0]])
        precision_points, recall_points, thresholds = [0], [0], [0]
        recall, precision = 0.0, 0.0

# PÁGINA 4: SIMULADOR DE RISCO
elif selected_page == "⚙️ Simulador de Risco":
    st.title("⚙️ Simulador de Risco de Crédito")
    
    # Função para preparar dados de entrada
    def prepare_input(form_data):
        input_df = pd.DataFrame([form_data])
        
        # Feature engineering (idêntico ao treinamento)
        input_df['Risco_Atrasos'] = input_df['Atrasos'] * input_df['Negativos']
        input_df['Historico_Risco'] = input_df['TempoCliente'] / (input_df['Atrasos'] + 1e-6)
        input_df['Alavancagem'] = input_df['Empréstimo'] / (input_df['ValorDoBem'] + 0.001)
        
        # One-hot encoding manual
        for cat in dados['Finalidade'].unique():
            input_df[f'Finalidade_{cat}'] = (input_df['Finalidade'] == cat).astype(int)
        
        for cat in dados['Emprego'].unique():
            input_df[f'Emprego_{cat}'] = (input_df['Emprego'] == cat).astype(int)
        
        # Remover colunas originais
        return input_df.drop(['Finalidade', 'Emprego'], axis=1)

    # Interface do formulário
    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            finalidade = st.selectbox("Finalidade", dados['Finalidade'].unique())
            emprego = st.selectbox("Emprego", dados['Emprego'].unique())
            emprestimo = st.number_input("Valor do Empréstimo (R$)", 
                                       min_value=1000, value=50000, step=1000)
            valor_bem = st.number_input("Valor do Bem (R$)", 
                                      min_value=1000, value=100000, step=1000)
            
        with col2:
            atrasos = st.number_input("Atrasos", min_value=0, value=0)
            negativos = st.number_input("Negativos", min_value=0, value=0)
            tempo_cliente = st.number_input("Tempo como Cliente (meses)", 
                                          min_value=1, value=120)
            rds = st.slider("Renda Comprometida (RDS %)", 0.0, 100.0, 30.0)
        
        submitted = st.form_submit_button("Calcular Risco")
        
        if submitted:
            form_data = {
                'Finalidade': finalidade,
                'Emprego': emprego,
                'Empréstimo': emprestimo,
                'ValorDoBem': valor_bem,
                'Atrasos': atrasos,
                'Negativos': negativos,
                'TempoCliente': tempo_cliente,
                'RDS': rds,
                'LC-Recente': 0,  # Valores padrão
                'LC-Atual': 0
            }
            
            try:
                input_data = prepare_input(form_data)
                
                # Verificação final antes da previsão
                if not hasattr(model, 'predict_proba'):
                    st.error("O modelo não possui método predict_proba()")
                    st.stop()
                
                proba = model.predict_proba(input_data)[0][1]
                
                # Visualização dos resultados
                st.success(f"Probabilidade de mau pagador: {proba:.2%}")
                
                # Gauge de probabilidade
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba*100,
                    title="Probabilidade de Inadimplência",
                    gauge={
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': OPTIMAL_THRESHOLD * 100
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Classificação final
                if proba >= OPTIMAL_THRESHOLD:
                    st.error("Classificação: Mau Pagador (risco alto)")
                else:
                    st.success("Classificação: Bom Pagador (risco baixo)")
                
            except Exception as e:
                st.error(f"Erro na previsão: {str(e)}")
                st.write("Dados enviados:", input_data)

# PÁGINA 5: IMPACTO NO NEGÓCIO
elif selected_page == "💼 Impacto no Negócio":
    st.title("💼 Calculadora de Impacto Financeiro")
    st.markdown("Estime o valor financeiro que o modelo pode economizar para a empresa.")

    try:
        X_raw = dados.drop('Cliente', axis=1)
        y_true = dados['Cliente'].map({'bom pagador': 0, 'mau pagador': 1})
        y_proba = model.predict_proba(X_raw)[:, 1]
        y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        fn_reais = cm[1][0]
    except Exception:
        fn_reais = 0

    st.subheader("Parâmetros da Simulação")
    col1, col2, col3 = st.columns(3)
    with col1:
        valor_medio_emprestimo = st.number_input("Valor Médio por Empréstimo (R$)", min_value=1000, value=25000, step=1000)
    with col2:
        taxa_perda = st.slider("Taxa de Perda sobre Inadimplência (%)", 10, 100, 60)
    with col3:
        fn_evitados = st.number_input("Falsos Negativos Evitados", min_value=0, value=int(fn_reais), step=1)

    prejuizo_por_fn = valor_medio_emprestimo * (taxa_perda / 100)
    economia_total = fn_evitados * prejuizo_por_fn

    st.markdown("---")
    st.subheader("Resultados da Simulação")
    st.info(f"Considerando os **{fn_evitados} Falsos Negativos** que o modelo pode identificar:")

    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Prejuízo por Cliente Não Detectado", f"R$ {prejuizo_por_fn:,.2f}")
    col_res2.metric("Economia Total Potencial", f"R$ {economia_total:,.2f}")

    st.markdown("---")
    st.subheader("Otimização do Limiar de Decisão")
    st.markdown(f"""
    O modelo está usando um limiar de **{OPTIMAL_THRESHOLD:.2%}** para classificar maus pagadores.
    Ajustar esse limiar afeta o trade-off entre:
    - **Falsos Negativos** (maus pagadores classificados como bons - risco para o banco)
    - **Falsos Positivos** (bons pagadores classificados como maus - perda de receita)
    """)

    novo_limiar = st.slider("Teste diferentes limiares", 0.0, 1.0, float(OPTIMAL_THRESHOLD), 0.01)

    if novo_limiar != OPTIMAL_THRESHOLD:
        try:
            y_pred_novo = (y_proba >= novo_limiar).astype(int)
            cm_novo = confusion_matrix(y_true, y_pred_novo)
            tn, fp, fn, tp = cm_novo.ravel()
            
            st.markdown(f"""
            **Resultados com limiar de {novo_limiar:.2%}:**
            - Falsos Negativos: **{fn}** (redução de {fn_reais - fn} em relação ao limiar atual)
            - Falsos Positivos: **{fp}** (aumento de {fp - cm[0][1]} em relação ao limiar atual)
            """)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Atual', 'Novo'],
                y=[fn_reais, fn],
                name='Falsos Negativos',
                marker_color='red'
            ))
            fig.add_trace(go.Bar(
                x=['Atual', 'Novo'],
                y=[cm[0][1], fp],
                name='Falsos Positivos',
                marker_color='orange'
            ))
            fig.update_layout(
                barmode='group',
                title='Impacto do Novo Limiar',
                yaxis_title='Quantidade'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível simular o novo limiar. Erro: {e}")
