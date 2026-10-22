# Challenge TOTVS — Inteligência de Reuniões

Protótipo Streamlit com foco em pessoas de negócio. A interface foi simplificada para três áreas: **Visão geral**, **Reuniões** e **Insights**. O notebook original continua sendo a referência acadêmica dos experimentos.

## O que a aplicação faz

- Resume o volume da base e as notas NPS disponíveis.
- Permite buscar reuniões por ID, segmento, UF ou trecho da transcrição.
- Exibe sinais exploratórios de sentimento, atenção e oportunidade depois de preparar o modelo.
- Resume temas recorrentes e produtos mencionados por dicionários de termos.
- Aceita JSON, JSONL e NDJSON; a base pode ser enviada pela interface e não precisa ficar no GitHub.

## Executar localmente

Requer Python 3.11 ou 3.12.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Publicar no Streamlit Community Cloud

1. Envie `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `Procfile` e a pasta `.streamlit/` ao GitHub.
2. Não envie `__pycache__/`, ambientes virtuais, credenciais ou bases de transcrições reais.
3. Acesse https://share.streamlit.io/ e entre com o GitHub.
4. Crie um app escolhendo o repositório, a branch `main` e `app.py` como arquivo principal.
5. Clique em **Deploy** e aguarde a instalação das dependências.

O deploy precisa ser confirmado por você na plataforma. O endereço público será gerado pelo Streamlit Community Cloud.

## Formato da base

A coluna obrigatória é `ANON_TRANSCRICAO`. Para treinar o classificador, a coluna `NOTA_NPS` também é necessária. `ID_MEETING` é opcional. Metadados como `NOME_SEGMENTO`, `UF`, `DURACAO_MEETING` e `DT_MEETING` são usados quando disponíveis.

O notebook usa NPS 0–6 como negativo e 9–10 como positivo. Notas 7–8 ficam fora do treino binário.

## Limitações e privacidade

O classificador usa TF-IDF + Regressão Logística, seguindo o baseline do notebook. Optuna e o self-training completo não são executados automaticamente no site. Os sinais de atenção e oportunidade são exploratórios, relativos à base carregada, e não são probabilidades calibradas de churn ou venda. Temas e produtos são identificados por dicionários simples; os resultados precisam de revisão humana.

O aplicativo público não deve receber transcrições confidenciais ou dados pessoais sem autorização e avaliação de privacidade. Não publique a base real no GitHub.
