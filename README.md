# Ingestão e Busca Semântica com LangChain e Postgres

CLI de perguntas e respostas sobre um PDF. A ingestão quebra o documento em chunks, gera embeddings e grava no
PostgreSQL com pgVector; o chat vetoriza a pergunta, busca os 10 trechos mais próximos e responde **somente** com
base neles.

Funciona com OpenAI ou Gemini — a escolha é feita por variável de ambiente, sem mexer no código.

## Requisitos

- Python 3.12 ou 3.13 (as dependências de `requirements.txt` ainda não têm wheels para 3.14)
- Docker e Docker Compose
- Uma API key da OpenAI **ou** do Google

## Configuração

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Preencha o `.env` conforme o provedor escolhido:

| Variável | Descrição | Padrão |
|---|---|---|
| `LLM_PROVIDER` | `openai` ou `google` | `openai` |
| `OPENAI_API_KEY` | Obrigatória quando `LLM_PROVIDER=openai` | — |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings da OpenAI | `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL` | Modelo de resposta da OpenAI | `gpt-4o-mini` |
| `GOOGLE_API_KEY` | Obrigatória quando `LLM_PROVIDER=google` | — |
| `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings do Gemini | `models/gemini-embedding-001` |
| `GOOGLE_CHAT_MODEL` | Modelo de resposta do Gemini | `gemini-2.5-flash-lite` |
| `DATABASE_URL` | Conexão do Postgres (driver `psycopg`) | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector | `documentos` |
| `PDF_PATH` | Caminho do PDF; relativo resolve a partir da raiz do projeto | `document.pdf` |

## Ordem de execução

1. Subir o banco:

```bash
docker compose up -d
```

2. Executar a ingestão do PDF:

```bash
python src/ingest.py
```

3. Rodar o chat:

```bash
python src/chat.py
```

## Uso

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair` (ou `Ctrl+D`) para encerrar.

Se a chamada ao provedor falhar no meio da conversa — limite do plano gratuito, chave inválida, rede — o chat mostra
o erro e continua esperando a próxima pergunta. Com o banco fora do ar, ele pede para subir o Postgres com
`docker compose up -d`.

## Como funciona

| Arquivo | Responsabilidade |
|---|---|
| `src/providers.py` | Lê e valida o `.env`, monta embeddings, LLM e o vector store conforme `LLM_PROVIDER` |
| `src/ingest.py` | `PyPDFLoader` → chunks de 1000 caracteres com overlap de 150 → embeddings → pgVector |
| `src/search.py` | `similarity_search_with_score(pergunta, k=10)` → contexto → prompt → LLM |
| `src/chat.py` | Loop de perguntas e respostas no terminal |

Cada chunk é gravado com um id determinístico (`<arquivo>:<índice>`), então rodar a ingestão de novo **atualiza**
os mesmos registros em vez de duplicar o documento.

O prompt restringe a resposta ao contexto recuperado: quando a informação não está no PDF, a resposta é sempre
`Não tenho informações necessárias para responder sua pergunta.`

## Trocando de provedor ou de modelo de embeddings

A tabela de vetores é criada na primeira ingestão com a dimensão do modelo escolhido (`text-embedding-3-small` gera
1536 dimensões; `gemini-embedding-001` gera 3072). Ao trocar o modelo de embeddings, apague a collection antiga antes
de reingerir:

```bash
docker compose down -v && docker compose up -d
python src/ingest.py
```

## Testes

```bash
pip install -r requirements-dev.txt
pytest          # testes unitários, sem rede e sem banco
ruff check .    # lint
mypy            # type-check
```
