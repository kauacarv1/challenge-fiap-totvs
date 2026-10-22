import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
TEXT_COLUMN = "ANON_TRANSCRICAO"
TARGET_COLUMN = "NOTA_NPS"
LABEL2ID = {"Negativo": 0, "Positivo": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NOTA_LIMIAR_DETRATOR = 6
NOTA_LIMIAR_PROMOTOR = 9

st.set_page_config(page_title="TOTVS | Inteligência de Reuniões", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1440px;}
[data-testid="stMetric"] {background: rgba(120,120,120,.07); border: 1px solid rgba(120,120,120,.16); padding: 16px; border-radius: 12px;}
div[data-testid="stTabs"] button {font-weight: 600;}
.small-note {color: #777; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

PRODUTOS = {
    "TOTVS RM (gestão/ERP)": ["rm"],
    "Fluig (processos/documentos)": ["fluig"],
    "Datasul / Logix / Protheus (ERP)": ["datasul", "logix", "protheus", "microsiga"],
    "Saúde (hospitalar/clínicas)": ["hospitalar", "prontuário", "tss", "glosa", "operadora", "convênio", "clínica", "laboratório"],
    "Backoffice": ["backoffice"], "RD Station / CRM": ["rd station", "crm"],
    "Techfin (financeiro/pagamentos)": ["techfin", "pix", "conciliação", "cobrança", "tesouraria"],
    "BI / Analytics": ["bi", "analytics", "smart view"],
    "Educacional": ["educacional", "matrícula", "aluno", "escola"],
    "Varejo / PDV": ["pdv", "loja", "supermercado"],
    "Gestão de Pessoas (folha/HCM)": ["folha", "gestão de pessoas", "hcm"],
    "IA / Agentes": ["inteligência artificial", "chatbot", "agente", "assistente"],
}
TERMOS_ATRITO = ["cancelar", "cancelamento", "concorrente", "insatisfeito", "reclamação", "problema", "problemas", "atraso", "demora", "erro", "erros", "falha", "instabilidade", "travando", "migrar", "trocar", "dificuldade", "chamado", "chamados", "parado"]
TERMOS_OPORTUNIDADE = ["proposta", "orçamento", "cotação", "módulo", "módulos", "licença", "licenças", "usuários", "expansão", "ampliar", "upgrade", "demonstração", "demo", "piloto", "interesse", "contratar", "contratação", "fechar", "fechamento", "assinatura", "desconto", "investimento", "roadmap"]
STOPWORDS_PT = set("a o as os de da do das dos em um uma uns umas e é ou para por com que se na no nas nos como mais mas ao aos à às já também muito foi ser está estão era eram foi foram são sua seu suas seus ele ela eles elas eu você vocês nós meu minha nosso nossa isso isto aquele aquela aqui ali lá então tá né tudo bem bom vai vou pode podem poderia seria ter tem tinha há haver quando onde qual quais porque pois até entre sobre sem após antes durante muito pouco cada todos todas mesmo ainda só não nao nem nunca jamais tampouco nenhum nenhuma nada ninguém".split())
NEGACOES_PRESERVADAS = {"não", "nao", "nem", "nunca", "jamais", "tampouco", "nenhum", "nenhuma", "nada", "ninguém"}
STOPWORDS_PT -= NEGACOES_PRESERVADAS
TAG_LOCUTOR = re.compile(r"\[LOCUTOR\s*\d+\]:\s*")
ESPACOS = re.compile(r"\s+")
PONTUACAO_REPETIDA = re.compile(r"([!?.,])\1{1,}")
TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*")


def limpar_transcricao(texto):
    texto = "" if texto is None else str(texto)
    texto = TAG_LOCUTOR.sub(" ", texto)
    texto = PONTUACAO_REPETIDA.sub(r"\1", texto)
    return ESPACOS.sub(" ", texto).strip()


def preparar_texto(texto):
    tokens = TOKEN_PATTERN.findall(limpar_transcricao(texto).lower())
    return " ".join(t for t in tokens if t not in STOPWORDS_PT and len(t) > 1)


def nota_para_polaridade(valor):
    if pd.isna(valor) or str(valor).strip().lower() in {"", "null", "none", "nan"}:
        return None
    try:
        nota = float(str(valor).replace(",", "."))
    except (ValueError, TypeError):
        return None
    if nota <= NOTA_LIMIAR_DETRATOR:
        return "Negativo"
    if nota >= NOTA_LIMIAR_PROMOTOR:
        return "Positivo"
    return None


def ler_arquivo(upload):
    nome = upload.name.lower()
    raw = upload.getvalue().decode("utf-8-sig")
    if nome.endswith(".jsonl") or nome.endswith(".ndjson"):
        rows = [json.loads(linha) for linha in raw.splitlines() if linha.strip()]
        return pd.DataFrame(rows)
    obj = json.loads(raw)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        for chave in ("data", "records", "items", "results"):
            if isinstance(obj.get(chave), list):
                return pd.DataFrame(obj[chave])
        return pd.DataFrame([obj])
    raise ValueError("Formato JSON não reconhecido.")


def compilar(termos):
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(termos, key=len, reverse=True)) + r")\b", re.IGNORECASE)

REGEX_PRODUTOS = {nome: compilar(termos) for nome, termos in PRODUTOS.items()}
REGEX_ATRITO = compilar(TERMOS_ATRITO)
REGEX_OPORTUNIDADE = compilar(TERMOS_OPORTUNIDADE)


def identificar_produtos(texto, top_n=2):
    contagem = {nome: len(regex.findall(texto.lower())) for nome, regex in REGEX_PRODUTOS.items()}
    ordenado = sorted(contagem.items(), key=lambda item: item[1], reverse=True)
    achados = [(nome, qtd) for nome, qtd in ordenado[:top_n] if qtd >= 3]
    return achados or [("Não identificado", 0)]


def densidade(regex, texto):
    return 1000 * len(regex.findall(texto.lower())) / max(len(texto.split()), 1)


def contar_termos(regex, texto, n=5):
    return Counter(regex.findall(texto.lower())).most_common(n)


def faixa(score, nomes=("Alto", "Médio", "Baixo")):
    if score >= 70:
        return nomes[0]
    if score >= 40:
        return nomes[1]
    return nomes[2]


def recomendar_acao(risco, oportunidade, produto, atrito):
    acoes = []
    nome_produto = f"de {produto}" if produto != "Não identificado" else "do produto discutido"
    atritos = ", ".join(t for t, _ in atrito[:3])
    if risco == "Alto":
        acoes.append("Acionar Customer Success em até 48h para uma conversa de retenção com o gestor do cliente.")
        if atritos:
            acoes.append(f"Levantar as pendências ligadas a: {atritos}.")
        if oportunidade == "Alta":
            acoes.append("Há interesse comercial, mas priorize resolver pendências antes de enviar uma nova proposta.")
    elif risco == "Médio":
        acoes.append("Fazer contato de acompanhamento em até 15 dias para confirmar a satisfação.")
        if oportunidade in ("Alta", "Média"):
            acoes.append(f"Retomar o assunto comercial {nome_produto} com o cliente.")
    elif oportunidade == "Alta":
        acoes.append(f"Priorizar o envio de proposta ou nova demonstração {nome_produto}.")
    elif oportunidade == "Média":
        acoes.append(f"Manter cadência de contato e apresentar evoluções {nome_produto}.")
    else:
        acoes.append("Manter o relacionamento de rotina e acompanhar o próximo NPS.")
    return acoes


def analisar_transcricao(texto, modelo, ref_neg, ref_pos, ref_oport, termos_desc):
    limpo = limpar_transcricao(texto)
    tokens = preparar_texto(limpo)
    if not tokens.strip():
        raise ValueError("A transcrição não contém texto suficiente após a limpeza.")
    probas = modelo.predict_proba([tokens])[0]
    classes = list(modelo.classes_)
    p_neg = float(probas[classes.index(0)])
    p_pos = float(probas[classes.index(1)])
    risco = 100 * np.searchsorted(ref_neg, p_neg, side="right") / max(len(ref_neg), 1)
    sentimento_score = 100 * np.searchsorted(ref_pos, p_pos, side="right") / max(len(ref_pos), 1)
    oport_dens = densidade(REGEX_OPORTUNIDADE, limpo)
    score_oport = 0.6 * (100 * np.searchsorted(ref_oport, oport_dens, side="right") / max(len(ref_oport), 1)) + 0.4 * sentimento_score
    sentimento = "Positivo" if sentimento_score >= 70 else ("Negativo" if sentimento_score < 30 else "Misto")
    produtos = identificar_produtos(limpo)
    atrito = contar_termos(REGEX_ATRITO, limpo)
    oportunidade_termos = contar_termos(REGEX_OPORTUNIDADE, limpo)
    risco_faixa = faixa(risco)
    oportunidade_faixa = faixa(score_oport, ("Alta", "Média", "Baixa"))
    termos = sorted(((termo, tokens.split().count(termo)) for termo in set(tokens.split())), key=lambda x: x[1], reverse=True)[:8]
    recomendacao = recomendar_acao(risco_faixa, oportunidade_faixa, produtos[0][0], atrito)
    return {"produto": produtos[0][0], "produtos": produtos, "sentimento": sentimento,
            "p_positivo": round(p_pos, 3), "p_negativo": round(p_neg, 3),
            "risco_churn_score": round(float(risco), 1), "risco_churn_faixa": risco_faixa,
            "oportunidade_score": round(float(score_oport), 1), "oportunidade_faixa": oportunidade_faixa,
            "termos_principais": [t for t, _ in termos], "termos_atrito": atrito,
            "termos_oportunidade": oportunidade_termos, "recomendacao": recomendacao}


def analisar_lote(df, modelo, refs, termos_desc):
    linhas = []
    for idx, linha in df.iterrows():
        texto = linha.get(TEXT_COLUMN, "")
        try:
            r = analisar_transcricao(texto, modelo, *refs, termos_desc)
            linhas.append({"ID_MEETING": linha.get("ID_MEETING", idx), "produto": r["produto"],
                           "sentimento": r["sentimento"], "p_positivo": r["p_positivo"],
                           "risco_churn_score": r["risco_churn_score"], "risco_churn_faixa": r["risco_churn_faixa"],
                           "oportunidade_score": r["oportunidade_score"], "oportunidade_faixa": r["oportunidade_faixa"],
                           "termos_principais": ", ".join(r["termos_principais"]),
                           "termos_atrito": ", ".join(t for t, _ in r["termos_atrito"]),
                           "recomendacao": " ".join(r["recomendacao"])})
        except ValueError:
            continue
    return pd.DataFrame(linhas)


def treinar_modelo(df, min_df, max_features, test_size):
    base = df.copy()
    base["_nota_num"] = pd.to_numeric(base[TARGET_COLUMN].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    base["_classe"] = base[TARGET_COLUMN].apply(nota_para_polaridade)
    base = base[base["_classe"].notna()].copy()
    base["_tokens"] = base[TEXT_COLUMN].apply(preparar_texto)
    base = base[base["_tokens"].str.len() > 0]
    contagem = base["_classe"].value_counts()
    if len(contagem) < 2 or contagem.min() < 3:
        raise ValueError("São necessários pelo menos 3 exemplos de cada classe: NPS 0–6 (Negativo) e 9–10 (Positivo).")
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        base["_tokens"], base["_classe"].map(LABEL2ID), test_size=test_size,
        stratify=base["_classe"], random_state=SEED)
    min_df_real = min(min_df, max(1, int(len(X_treino) * 0.05)))
    modelo = Pipeline([("tfidf", TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None,
                                                  max_features=max_features, min_df=min_df_real, sublinear_tf=True)),
                       ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=SEED))])
    modelo.fit(X_treino, y_treino)
    pred = modelo.predict(X_teste)
    prob = modelo.predict_proba(X_teste)[:, list(modelo.classes_).index(1)]
    metricas = {"accuracy": accuracy_score(y_teste, pred), "f1_macro": f1_score(y_teste, pred, average="macro"),
                "recall_negativo": recall_score(y_teste, pred, pos_label=0, zero_division=0),
                "precision_macro": precision_score(y_teste, pred, average="macro", zero_division=0),
                "auc": roc_auc_score((np.array(y_teste) == 1).astype(int), prob) if len(set(y_teste)) == 2 else np.nan,
                "y_teste": np.array(y_teste), "pred": pred, "relatorio": classification_report(y_teste, pred, labels=[0, 1], target_names=["Negativo", "Positivo"], output_dict=True, zero_division=0),
                "n_rotulados": len(base), "n_treino": len(X_treino), "n_teste": len(X_teste), "contagem": contagem.to_dict()}
    modelo_final = Pipeline([("tfidf", TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None,
                                                        max_features=max_features, min_df=min_df_real, sublinear_tf=True)),
                             ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=SEED))])
    modelo_final.fit(base["_tokens"], base["_classe"].map(LABEL2ID))
    nao_rotulados = df[df[TARGET_COLUMN].apply(nota_para_polaridade).isna()].copy()
    neutros = df[df[TARGET_COLUMN].apply(lambda v: pd.notna(v) and str(v).strip() not in {"", "null", "nan", "None"}) & df[TARGET_COLUMN].apply(lambda v: nota_para_polaridade(v) is None)].copy()
    ref_df = pd.concat([neutros, nao_rotulados], ignore_index=True)
    if ref_df.empty:
        ref_df = df.copy()
    ref_tokens = ref_df[TEXT_COLUMN].fillna("").apply(preparar_texto)
    ref_tokens = ref_tokens[ref_tokens.str.len() > 0]
    ref_probs = modelo_final.predict_proba(ref_tokens)
    pos_idx = list(modelo_final.classes_).index(1)
    neg_idx = list(modelo_final.classes_).index(0)
    ref_neg = np.sort(ref_probs[:, neg_idx]) if len(ref_probs) else np.array([0.0, 1.0])
    ref_pos = np.sort(ref_probs[:, pos_idx]) if len(ref_probs) else np.array([0.0, 1.0])
    ref_oport = np.sort(np.array([densidade(REGEX_OPORTUNIDADE, limpar_transcricao(t)) for t in ref_df[TEXT_COLUMN].fillna("")]))
    if len(ref_oport) == 0:
        ref_oport = np.array([0.0, 1.0])
    terms_desc = modelo_final.named_steps["tfidf"].get_feature_names_out()
    return modelo_final, metricas, (ref_neg, ref_pos, ref_oport), terms_desc


def carregar_demo():
    return pd.DataFrame([
        {"ID_MEETING": "DEMO-001", "DT_MEETING": "2026-01-10", "ANON_TRANSCRICAO": "[LOCUTOR 1]: Estamos com problema de instabilidade e chamados sem retorno. O sistema apresenta erro e demora para resolver.", "NOTA_NPS": 4, "NOME_SEGMENTO": "SERVICOS", "UF": "SP", "DURACAO_MEETING": "00:30:00"},
        {"ID_MEETING": "DEMO-002", "DT_MEETING": "2026-01-11", "ANON_TRANSCRICAO": "[LOCUTOR 1]: Estamos satisfeitos com a solução, o atendimento foi ótimo e queremos conhecer novos módulos. Gostaríamos de uma demonstração e proposta.", "NOTA_NPS": 10, "NOME_SEGMENTO": "MANUFATURA", "UF": "SP", "DURACAO_MEETING": "00:25:00"},
        {"ID_MEETING": "DEMO-003", "DT_MEETING": "2026-01-12", "ANON_TRANSCRICAO": "[LOCUTOR 1]: Queremos conversar sobre orçamento, licença adicional, upgrade e demonstração para mais usuários.", "NOTA_NPS": None, "NOME_SEGMENTO": "VAREJO", "UF": "RJ", "DURACAO_MEETING": "00:20:00"},
        {"ID_MEETING": "DEMO-004", "DT_MEETING": "2026-01-13", "ANON_TRANSCRICAO": "[LOCUTOR 1]: O Protheus está com problemas e falhas, mas o time de suporte ajudou. Vamos avaliar uma proposta para o próximo trimestre.", "NOTA_NPS": 6, "NOME_SEGMENTO": "MANUFATURA", "UF": "MG", "DURACAO_MEETING": "00:35:00"},
        {"ID_MEETING": "DEMO-005", "DT_MEETING": "2026-01-14", "ANON_TRANSCRICAO": "[LOCUTOR 1]: A equipe está satisfeita, queremos avaliar o Fluig e conhecer possibilidades de expansão.", "NOTA_NPS": 9, "NOME_SEGMENTO": "SERVICOS", "UF": "PR", "DURACAO_MEETING": "00:18:00"},
        {"ID_MEETING": "DEMO-006", "DT_MEETING": "2026-01-15", "ANON_TRANSCRICAO": "[LOCUTOR 1]: Ainda temos dúvidas sobre o sistema, precisamos de mais treinamento e vamos acompanhar os chamados abertos.", "NOTA_NPS": None, "NOME_SEGMENTO": "SAUDE", "UF": "SP", "DURACAO_MEETING": "00:22:00"},
        {"ID_MEETING": "DEMO-007", "DT_MEETING": "2026-01-16", "ANON_TRANSCRICAO": "O atendimento demorou e houve erro na integração. Precisamos resolver os problemas antes de renovar.", "NOTA_NPS": 3, "NOME_SEGMENTO": "SAUDE", "UF": "RJ", "DURACAO_MEETING": "00:28:00"},
        {"ID_MEETING": "DEMO-008", "DT_MEETING": "2026-01-17", "ANON_TRANSCRICAO": "Estamos muito satisfeitos com o suporte e a solução atende nossas necessidades. Queremos ampliar o uso.", "NOTA_NPS": 10, "NOME_SEGMENTO": "SERVICOS", "UF": "SP", "DURACAO_MEETING": "00:21:00"},
        {"ID_MEETING": "DEMO-009", "DT_MEETING": "2026-01-18", "ANON_TRANSCRICAO": "Tivemos falhas e instabilidade, além de chamados sem retorno. A experiência precisa melhorar.", "NOTA_NPS": 5, "NOME_SEGMENTO": "VAREJO", "UF": "MG", "DURACAO_MEETING": "00:32:00"},
        {"ID_MEETING": "DEMO-010", "DT_MEETING": "2026-01-19", "ANON_TRANSCRICAO": "O projeto foi ótimo, a equipe é atenciosa e estamos satisfeitos. Gostaríamos de uma demonstração de novos módulos.", "NOTA_NPS": 9, "NOME_SEGMENTO": "EDUCACIONAL", "UF": "PR", "DURACAO_MEETING": "00:26:00"},
        {"ID_MEETING": "DEMO-011", "DT_MEETING": "2026-01-20", "ANON_TRANSCRICAO": "A solução apresenta problemas frequentes e demora no suporte. Estamos considerando trocar de sistema.", "NOTA_NPS": 2, "NOME_SEGMENTO": "MANUFATURA", "UF": "SP", "DURACAO_MEETING": "00:40:00"},
        {"ID_MEETING": "DEMO-012", "DT_MEETING": "2026-01-21", "ANON_TRANSCRICAO": "O atendimento foi excelente e a implantação correu bem. Queremos avaliar uma proposta para novas licenças.", "NOTA_NPS": 10, "NOME_SEGMENTO": "SERVICOS", "UF": "SC", "DURACAO_MEETING": "00:24:00"},
    ])

st.markdown("""
<style>
:root { --ink: #172033; --muted: #687386; --line: #e7eaf0; --accent: #d94b35; }
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
#MainMenu, footer {visibility: hidden;}
[data-testid="stMetric"] {background:#fff; border:1px solid var(--line); padding:16px 18px; border-radius:14px;}
[data-testid="stMetricLabel"] p {color:#687386; font-size:.85rem;}
[data-testid="stMetricValue"] {color:#172033;}
div[data-testid="stTabs"] button {font-weight:600;}
.hero {padding: 1.2rem 0 1.5rem 0; border-bottom:1px solid #e7eaf0; margin-bottom:1.2rem;}
.hero-kicker {font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; color:#d94b35; font-weight:700;}
.hero-title {font-size:2.1rem; line-height:1.15; font-weight:700; color:#172033; margin:.35rem 0;}
.hero-sub {font-size:1rem; color:#687386;}
.section-note {color:#687386; font-size:.9rem;}
.signal {border:1px solid #e7eaf0; border-radius:12px; padding:14px 16px; margin:.4rem 0;}
</style>
<div class="hero">
  <div class="hero-kicker">TOTVS · Customer intelligence</div>
  <div class="hero-title">Entenda o que acontece nas reuniões.</div>
  <div class="hero-sub">Sinais de atenção, temas recorrentes e próximos passos, em um só lugar.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Dados")
    arquivo = st.file_uploader("Carregar transcrições", type=["json", "jsonl", "ndjson"])
    usar_demo = st.checkbox("Visualizar exemplo", value=arquivo is None)
    st.caption("A base é processada nesta sessão. Não envie informações confidenciais para uma aplicação pública.")
    st.divider()
    st.markdown("### Configuração da análise")
    test_size = st.select_slider("Dados reservados para avaliação", options=[0.15, 0.20, 0.25, 0.30], value=0.20)
    max_features = st.select_slider("Detalhamento do vocabulário", options=[500, 1000, 2000, 3000, 5000], value=2000)
    min_df = st.select_slider("Frequência mínima de termos", options=[1, 2, 3, 4, 5], value=2)

if arquivo is not None:
    try:
        df = ler_arquivo(arquivo)
        st.sidebar.success(f"{len(df):,} reuniões carregadas")
    except Exception as exc:
        st.error(f"Não foi possível ler o arquivo: {exc}")
        st.stop()
elif usar_demo:
    df = carregar_demo()
    st.info("Modo demonstração: os exemplos são fictícios e servem apenas para conhecer a interface.")
else:
    st.info("Carregue um arquivo JSON/JSONL na barra lateral para começar.")
    st.stop()

if TEXT_COLUMN not in df.columns:
    st.error(f"O arquivo precisa conter a coluna {TEXT_COLUMN}.")
    st.stop()
if TARGET_COLUMN not in df.columns:
    df[TARGET_COLUMN] = None
if "ID_MEETING" not in df.columns:
    df["ID_MEETING"] = [f"REG-{i+1}" for i in range(len(df))]
for col in ["NOME_SEGMENTO", "UF", "FORMATO_MEETING", "STATUS_MEETING", "DURACAO_MEETING", "DT_MEETING"]:
    if col not in df.columns:
        df[col] = np.nan

df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
df["_nota_num"] = pd.to_numeric(df[TARGET_COLUMN].astype(str).str.replace(",", ".", regex=False), errors="coerce")
df["_classe"] = df[TARGET_COLUMN].apply(nota_para_polaridade)
df["_texto_limpo"] = df[TEXT_COLUMN].apply(limpar_transcricao)
df["_palavras"] = df["_texto_limpo"].str.split().str.len()
rotulados = df[df["_classe"].notna()].copy()
nao_rotulados = df[df["_classe"].isna() & df["_nota_num"].isna()].copy()

# Treinamento explícito, reutilizado pelas duas telas que precisam de inferência.
fingerprint = (len(df), int(df[TEXT_COLUMN].str.len().sum()), max_features, min_df, test_size)
pronto = "modelo" in st.session_state and st.session_state.get("fingerprint") == fingerprint
if not pronto and len(rotulados) >= 6:
    with st.expander("Preparar análise inteligente", expanded=False):
        st.write("Para classificar novas reuniões, o aplicativo precisa aprender com as notas NPS disponíveis nesta base.")
        if st.button("Preparar modelo", type="primary"):
            with st.spinner("Preparando a análise..."):
                try:
                    modelo, metricas, refs, termos_desc = treinar_modelo(df, min_df, max_features, test_size)
                    st.session_state["modelo"] = modelo
                    st.session_state["metricas"] = metricas
                    st.session_state["refs"] = refs
                    st.session_state["termos_desc"] = termos_desc
                    st.session_state["fingerprint"] = fingerprint
                    st.rerun()
                except Exception as exc:
                    st.error(f"Não foi possível preparar o modelo: {exc}")

visao, reunioes, insights = st.tabs(["Visão geral", "Reuniões", "Insights"])
with visao:
    st.subheader("Resumo da carteira")
    total = len(df)
    com_texto = int((df[TEXT_COLUMN].str.strip().str.len() > 0).sum())
    com_nps = int(df["_nota_num"].between(0, 10).sum())
    negativos = int((df["_nota_num"] <= 6).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reuniões", f"{total:,}")
    c2.metric("Com transcrição", f"{com_texto:,}")
    c3.metric("Com nota NPS", f"{com_nps:,}")
    c4.metric("Sinais de insatisfação no NPS", f"{negativos:,}")

    esquerda, direita = st.columns([1.1, 1])
    with esquerda:
        st.markdown("#### Satisfação registrada")
        notas = df[df["_nota_num"].between(0, 10)]["_nota_num"].value_counts().sort_index().rename_axis("Nota NPS").reset_index(name="Reuniões")
        if notas.empty:
            st.info("Não há notas NPS válidas nesta base.")
        else:
            fig = px.bar(notas, x="Nota NPS", y="Reuniões", color_discrete_sequence=["#d94b35"])
            fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300, xaxis=dict(dtick=1), yaxis_title="Reuniões")
            st.plotly_chart(fig, use_container_width=True)
    with direita:
        st.markdown("#### Onde estão as conversas")
        segmentos = df["NOME_SEGMENTO"].fillna("Não informado").value_counts().head(7).rename_axis("Segmento").reset_index(name="Reuniões")
        fig = px.bar(segmentos, x="Reuniões", y="Segmento", orientation="h", color_discrete_sequence=["#263c63"])
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300, yaxis={"categoryorder":"total ascending"}, xaxis_title="Reuniões")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Ponto de atenção")
    if negativos:
        st.markdown(f'<div class="signal"><b>{negativos:,} reunião(ões) com NPS de detrator</b><br><span class="section-note">Notas de 0 a 6. Vale revisar os relatos relacionados antes de definir qualquer ação.</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal"><b>Nenhum detrator identificado pelas notas disponíveis</b><br><span class="section-note">Isso não significa ausência de problemas: muitas reuniões podem não ter avaliação NPS.</span></div>', unsafe_allow_html=True)
    st.caption("Os dados refletem apenas o arquivo carregado. Sinais extraídos de texto são indicativos e devem ser revisados por uma pessoa.")

with reunioes:
    st.subheader("Encontre uma reunião")
    busca = st.text_input("Buscar por ID, segmento ou trecho da transcrição", placeholder="Ex.: integração, suporte, DEMO-001")
    colf1, colf2 = st.columns(2)
    segmentos_disponiveis = sorted(df["NOME_SEGMENTO"].dropna().astype(str).unique())
    ufs_disponiveis = sorted(df["UF"].dropna().astype(str).unique())
    with colf1:
        segmento_sel = st.multiselect("Segmento", segmentos_disponiveis)
    with colf2:
        uf_sel = st.multiselect("UF", ufs_disponiveis)
    filtrado = df.copy()
    if segmento_sel:
        filtrado = filtrado[filtrado["NOME_SEGMENTO"].astype(str).isin(segmento_sel)]
    if uf_sel:
        filtrado = filtrado[filtrado["UF"].astype(str).isin(uf_sel)]
    if busca.strip():
        termo = busca.strip().lower()
        mascara = filtrado["ID_MEETING"].astype(str).str.lower().str.contains(termo, na=False)
        mascara |= filtrado["NOME_SEGMENTO"].astype(str).str.lower().str.contains(termo, na=False)
        mascara |= filtrado[TEXT_COLUMN].str.lower().str.contains(re.escape(termo), na=False)
        filtrado = filtrado[mascara]
    exibicao = filtrado[["ID_MEETING", "DT_MEETING", "NOME_SEGMENTO", "UF", "NOTA_NPS", "DURACAO_MEETING"]].copy()
    exibicao.columns = ["Reunião", "Data", "Segmento", "UF", "NPS", "Duração"]
    st.caption(f"{len(filtrado):,} reunião(ões) encontrada(s). Selecione uma para ver os detalhes.")
    st.dataframe(exibicao.head(300), use_container_width=True, hide_index=True)
    opcoes = filtrado.index.tolist()
    if opcoes:
        selecionado = st.selectbox("Abrir reunião", opcoes, format_func=lambda i: f"{df.loc[i, 'ID_MEETING']} · {df.loc[i, 'NOME_SEGMENTO'] if pd.notna(df.loc[i, 'NOME_SEGMENTO']) else 'Segmento não informado'}")
        linha = df.loc[selecionado]
        with st.expander("Ler transcrição", expanded=True):
            st.write(limpar_transcricao(linha[TEXT_COLUMN]))
        if st.session_state.get("fingerprint") == fingerprint and "modelo" in st.session_state:
            if st.button("Gerar resumo de sinais desta reunião"):
                try:
                    resultado = analisar_transcricao(linha[TEXT_COLUMN], st.session_state["modelo"], *st.session_state["refs"], st.session_state["termos_desc"])
                    st.session_state["reuniao_selecionada_analise"] = resultado
                except Exception as exc:
                    st.error(f"Não foi possível analisar esta reunião: {exc}")
            if "reuniao_selecionada_analise" in st.session_state:
                r = st.session_state["reuniao_selecionada_analise"]
                a, b, c = st.columns(3)
                a.metric("Tom estimado", r["sentimento"])
                b.metric("Sinal de atenção", r["risco_churn_faixa"])
                c.metric("Sinal comercial", r["oportunidade_faixa"])
                st.markdown("**Produto mencionado**")
                st.write(r["produto"])
                st.markdown("**Pontos para revisar**")
                st.write(", ".join(t for t, _ in r["termos_atrito"]) or "Nenhum termo de atrito do dicionário encontrado.")
                st.markdown("**Próximo passo sugerido**")
                for acao in r["recomendacao"]:
                    st.write(f"- {acao}")
        else:
            st.caption("Para gerar sinais automáticos, prepare o modelo no menu lateral.")

with insights:
    st.subheader("Temas que merecem atenção")
    st.write("Uma leitura exploratória dos assuntos citados nas conversas, para ajudar a encontrar padrões e perguntas para investigação.")
    todos_textos = " ".join(df["_texto_limpo"].tolist()).lower()
    atritos = contar_termos(REGEX_ATRITO, todos_textos, n=10)
    oportunidades = contar_termos(REGEX_OPORTUNIDADE, todos_textos, n=10)
    produtos = []
    for nome, termos in PRODUTOS.items():
        quantidade = sum(len(re.findall(r"\b" + re.escape(t) + r"\b", todos_textos)) for t in termos)
        if quantidade:
            produtos.append((nome, quantidade))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Dificuldades mencionadas")
        if atritos:
            fig = px.bar(pd.DataFrame(atritos, columns=["Tema", "Menções"]), x="Menções", y="Tema", orientation="h", color_discrete_sequence=["#d94b35"])
            fig.update_layout(margin=dict(l=0,r=0,t=5,b=0), height=330, yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhum termo do dicionário foi encontrado.")
    with col2:
        st.markdown("#### Assuntos comerciais")
        if oportunidades:
            fig = px.bar(pd.DataFrame(oportunidades, columns=["Tema", "Menções"]), x="Menções", y="Tema", orientation="h", color_discrete_sequence=["#263c63"])
            fig.update_layout(margin=dict(l=0,r=0,t=5,b=0), height=330, yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhum termo do dicionário foi encontrado.")
    st.markdown("#### Produtos e soluções citados")
    if produtos:
        produtos_df = pd.DataFrame(sorted(produtos, key=lambda x: x[1], reverse=True)[:10], columns=["Produto ou solução", "Menções"])
        fig = px.bar(produtos_df, x="Menções", y="Produto ou solução", orientation="h", color_discrete_sequence=["#526b8d"])
        fig.update_layout(margin=dict(l=0,r=0,t=5,b=0), height=330, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum produto do dicionário foi identificado nesta base.")
    st.caption("Os temas são contados por dicionários simples de palavras. Menções não representam, por si só, gravidade, intenção de compra ou risco real de cancelamento.")

with st.expander("Sobre esta análise"):
    st.markdown("""
- As notas NPS de 0 a 6 são tratadas como negativas e as notas 9 e 10 como positivas. Notas 7 e 8 ficam fora do treinamento binário.
- O modelo usa TF-IDF e Regressão Logística, seguindo o baseline do notebook acadêmico.
- Os sinais de atenção e oportunidade são indicadores exploratórios relativos à base carregada, não probabilidades calibradas de churn ou venda.
- A análise não substitui a leitura da conversa nem a validação da equipe responsável.
- Para manter o site leve, Optuna e o processo completo de self-training do notebook não são executados automaticamente.
    """)

st.divider()
st.caption("Challenge TOTVS · FIAP 2026 · Protótipo de apoio à leitura de reuniões")
