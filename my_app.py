# Arquivo Único: app_final.py
# ------------------------------------
# OBJETIVO: Versão final com pré-processamento manual para garantir compatibilidade com o modelo.
#
# Para rodá-lo, execute no seu terminal:
# streamlit run app_final.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, precision_recall_curve, classification_report
from plotly.figure_factory import create_annotated_heatmap

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard de Risco de Crédito",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

OPTIMAL_THRESHOLD = 0.42

# --- Funções de Carregamento ---
@st.cache_resource
def load_model(caminho_modelo):
    """Carrega o pipeline completo com ExtraTrees"""
    try:
        return joblib.load(caminho_modelo)
    except Exception as e:
        st.error(f"ERRO AO CARREGAR O MODELO: Verifique o nome do ficheiro e as dependências. Erro: {e}")
        return None

@st.cache_data
def load_data(caminho_dados):
    """Carrega os dados para a análise exploratória."""
    try:
        dados = pd.read_csv(caminho_dados)
        # Pré-processamento básico
        dados['RDS'] = dados['RDS'].astype(str).str.replace('%', '', regex=False)
        dados['RDS'] = pd.to_numeric(dados['RDS'], errors='coerce')
        return dados
    except Exception as e:
        st.error(f"ERRO AO CARREGAR OS DADOS: Verifique o nome do ficheiro '{caminho_dados}'. Erro: {e}")
        return None

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
    st.title("📊 Dashboard do Modelo de Risco")
    st.markdown("Visão geral do desempenho do modelo e da distribuição dos dados.")

    try:
        X_raw = dados.drop('Cliente', axis=1)
        y_true = dados['Cliente'].map({'bom pagador': 0, 'mau pagador': 1})
        y_proba = model.predict_proba(X_raw)[:, 1]
        y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
        
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
    except Exception as e:
        st.warning(f"Não foi possível calcular as métricas de performance. Verifique os dados e o modelo. Erro: {e}")
        recall, precision, accuracy = 0.0, 0.0, 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall do Modelo", f"{recall:.2%}", help="Capacidade de identificar os 'maus pagadores'.")
    col2.metric("Precisão do Modelo", f"{precision:.2%}", help="Assertividade do modelo ao classificar um cliente como 'mau pagador'.")
    col3.metric("Acurácia Geral", f"{accuracy:.2%}", help="Percentual geral de acertos do modelo.")
    col4.metric("Taxa de Inadimplência", f"{(dados_filtrados['Cliente'] == 'mau pagador').mean():.2%}", "Observada nos dados filtrados")

    st.markdown("---")
    st.subheader("Distribuição de Clientes")
    cliente_counts = dados_filtrados['Cliente'].value_counts()
    fig_pie = px.pie(values=cliente_counts.values, names=cliente_counts.index, title='Proporção de Bons vs. Maus Pagadores', hole=.3)
    st.plotly_chart(fig_pie, use_container_width=True)

# PÁGINA 2: ANÁLISE EXPLORATÓRIA
elif pagina == "📈 Análise Exploratória":
    st.title("📈 Análise Exploratória Interativa")
    st.markdown("Explore as relações entre as variáveis do conjunto de dados filtrado.")
    
    if dados is not None:
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
        cat_col_select = st.selectbox("Selecione uma variável categórica:", 
                                    options=dados_filtrados.select_dtypes(include='object').drop('Cliente', axis=1).columns)
        fig_bar = px.bar(dados_filtrados.groupby([cat_col_select, 'Cliente']).size().reset_index(name='count'), 
                        x=cat_col_select, y='count', color='Cliente', barmode='group', 
                        title=f'Contagem por {cat_col_select} e tipo de cliente')
        st.plotly_chart(fig_bar, use_container_width=True)

# PÁGINA 3: DETALHES DO MODELO
elif pagina == "🧠 Detalhes do Modelo":
    st.title("🧠 Análise Profunda do Modelo")
    st.markdown("Aqui exploramos o comportamento e a performance do modelo carregado.")
    
    try:
        X_raw = dados.drop('Cliente', axis=1)
        y_true = dados['Cliente'].map({'bom pagador': 0, 'mau pagador': 1})
        y_proba = model.predict_proba(X_raw)[:, 1]
        y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        precision_points, recall_points, thresholds = precision_recall_curve(y_true, y_proba)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
    except Exception as e:
        st.warning(f"Não foi possível gerar os gráficos de performance do modelo. Erro: {e}")
        cm = np.array([[0, 0], [0, 0]])
        precision_points, recall_points, thresholds = [0], [0], [0]
        recall, precision = 0.0, 0.0
    
    tab1, tab2, tab3 = st.tabs(["Matriz de Confusão", "Curvas de Performance", "Feature Importance"])
    
    with tab1:
        st.subheader("Matriz de Confusão")
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Previsto Bom', 'Previsto Mau'],
            y=['Real Bom', 'Real Mau'],
            colorscale='Greens',
            text=[[str(x) for x in row] for row in cm],
            texttemplate="%{text}",
            hoverongaps=False
        ))
        fig_cm.update_layout(title='Matriz de Confusão com Limiar Ajustado')
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Detalhes da matriz
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        - **Verdadeiros Positivos (TP):** {tp}
        - **Falsos Positivos (FP):** {fp}
        - **Falsos Negativos (FN):** {fn}
        - **Verdadeiros Negativos (TN):** {tn}
        """)
    
    with tab2:
        st.subheader("Curva Precision-Recall")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=recall_points,
            y=precision_points,
            mode='lines',
            name='Curva PR'
        ))
        fig_pr.add_shape(
            type='line',
            x0=0, x1=1, y0=1, y1=0,
            line=dict(color='RoyalBlue', width=2, dash='dot')
        )
        fig_pr.update_layout(
            title='Curva Precision-Recall',
            xaxis_title='Recall',
            yaxis_title='Precision'
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        
        st.subheader("Distribuição de Probabilidades")
        fig_dist = px.histogram(
            x=y_proba,
            color=y_true.map({0: 'Bom Pagador', 1: 'Mau Pagador'}),
            nbins=50,
            labels={'x': 'Probabilidade de Mau Pagador', 'color': 'Classe Real'},
            title='Distribuição das Probabilidades Previstas'
        )
        fig_dist.add_vline(x=OPTIMAL_THRESHOLD, line_dash="dash", line_color="red")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab3:
        st.subheader("Importância das Features")
        try:
            importances = model.named_steps['classifier'].feature_importances_
            feature_names = ['Empréstimo', 'ValorDoBem', 'TempoEmprego', 'Negativos', 
                           'Atrasos', 'TempoCliente', 'LC-Recente', 'LC-Atual', 'RDS',
                           'Risco_Atrasos', 'Historico_Risco', 'Alavancagem']
            
            # Adicionar nomes das features categóricas
            categorical_features = ['Emprego', 'Finalidade']
            ohe_features = model.named_steps['preprocessor'].transformers_[1][1]\
                .named_steps['onehot'].get_feature_names_out(categorical_features)
            feature_names.extend(ohe_features)
            
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 20 Features Mais Importantes'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível extrair a importância das features. Erro: {e}")

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
            alavancagem = emprestimo_valor / (valor_do_bem + 0.001)
            st.metric("Alavancagem", f"{alavancagem:.2f}")
            
        with col2:
            emprego = st.selectbox("Tipo de Emprego", options=dados['Emprego'].unique())
            tempo_emprego = st.slider("Tempo no Emprego (anos)", 0.0, 40.0, 5.0)
            tempo_cliente = st.slider("Tempo como Cliente (meses)", 1.0, 500.0, 120.0)
            atrasos = st.slider("Número de Atrasos", 0, 20, 0)
            historico_risco = tempo_cliente / (atrasos + 1)
            st.metric("Histórico de Risco", f"{historico_risco:.2f}")
            
        with col3:
            negativos = st.slider("Registros Negativos", 0, 10, 0)
            risco_atrasos = negativos * atrasos
            st.metric("Risco de Atrasos", risco_atrasos)
            rds = st.slider("Renda Comprometida (RDS %)", 0.0, 100.0, 30.0)
            lc_recente = st.slider("Linhas de Crédito Recentes", 0, 10, 1)
            lc_atual = st.slider("Linhas de Crédito Atuais", 0, 20, 5)

    if st.button("Analisar Risco do Cliente", type="primary"):
        input_data_dict = {
            'Empréstimo': [emprestimo_valor],
            'ValorDoBem': [valor_do_bem],
            'Finalidade': [finalidade],
            'Emprego': [emprego],
            'TempoEmprego': [tempo_emprego],
            'Negativos': [negativos],
            'Atrasos': [atrasos],
            'TempoCliente': [tempo_cliente],
            'LC-Recente': [lc_recente],
            'LC-Atual': [lc_atual],
            'RDS': [rds],
            'Risco_Atrasos': [risco_atrasos],
            'Historico_Risco': [historico_risco],
            'Alavancagem': [alavancagem]
        }
        
        input_data = pd.DataFrame(input_data_dict)
        
        try:
            # Obter probabilidades
            proba = model.predict_proba(input_data)[0]
            prob_mau_pagador = proba[1]
            
            # Aplicar limiar ótimo
            classificacao = "mau pagador" if prob_mau_pagador >= OPTIMAL_THRESHOLD else "bom pagador"
            
            # Visualização 1: Gauge de probabilidade
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_mau_pagador * 100,
                title={'text': "Probabilidade de Inadimplência (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
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
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Visualização 2: Fatores de risco
            fatores_risco = {
                'Fator': ['Alavancagem', 'Risco Atrasos', 'Histórico Risco', 'RDS'],
                'Valor': [alavancagem, risco_atrasos, historico_risco, rds],
                'Peso': [0.4, 0.3, 0.2, 0.1]  # Pesos exemplos (ajustar conforme importância real)
            }
            fig_fatores = px.bar(
                fatores_risco,
                x='Fator',
                y='Valor',
                color='Peso',
                title='Principais Fatores de Risco',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_fatores, use_container_width=True)
            
            # Resultado da classificação
            if classificacao == "mau pagador":
                st.error(f"""
                **ALERTA DE RISCO:** 
                - Classificação: **Mau Pagador** 
                - Probabilidade: {prob_mau_pagador:.2%}
                - Limiar de decisão: {OPTIMAL_THRESHOLD:.2%}
                
                **Fatores críticos:**
                {', '.join([f for f, v in zip(['Alavancagem', 'Risco Atrasos', 'Histórico Risco', 'RDS'], 
                                            [alavancagem, risco_atrasos, historico_risco, rds]) 
                          if v > np.percentile(dados[[f]], 75)])}
                """)
            else:
                st.success(f"""
                **BAIXO RISCO:** 
                - Classificação: **Bom Pagador** 
                - Probabilidade: {prob_mau_pagador:.2%}
                - Limiar de decisão: {OPTIMAL_THRESHOLD:.2%}
                
                **Pontos positivos:**
                {', '.join([f for f, v in zip(['Alavancagem', 'Risco Atrasos', 'Histórico Risco', 'RDS'], 
                                            [alavancagem, risco_atrasos, historico_risco, rds]) 
                          if v < np.percentile(dados[[f]], 25)])}
                """)
                
        except Exception as e:
            st.error(f"Erro ao processar a análise de risco: {str(e)}")

# PÁGINA 5: IMPACTO NO NEGÓCIO
elif pagina == "💼 Impacto no Negócio":
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
