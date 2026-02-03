# Classificador de Sons Respiratórios (API)

Resumo rápido
- API em FastAPI que recebe um arquivo de áudio (`file` em `multipart/form-data`) no endpoint `POST /predict/` e retorna a classe prevista e confiança.

Dependências principais
- Python packages (ver `requirements.txt`): `fastapi`, `uvicorn`, `gunicorn`, `python-multipart`, `requests`, `boto3`, `tensorflow`, `librosa`, `scikit-learn`, etc.

Executando localmente (desenvolvimento)
1. Crie e ative virtualenv.
2. Instale dependências:

```bash
pip install -r requirements.txt
```

3. Rodar a API localmente:

```bash
py main.py
# ou
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Variáveis de ambiente para carregar o modelo (duas opções)

Opção A — Presigned URLs (recomendado, sem credenciais no servidor):
- `MODEL_S3_PRESIGNED_URL`  -> URL para `model.weights.h5`
- `MODEL_CONFIG_S3_PRESIGNED_URL` -> URL para `config.json`
- `ENCODER_S3_PRESIGNED_URL` -> URL para `label_encoder.pkl`

Opção B — Acesso S3 via boto3 (credenciais necessárias no ambiente):
- `S3_BUCKET` — nome do bucket
- `MODEL_S3_KEY` — chave do objeto para `model.weights.h5`
- `CONFIG_S3_KEY` — chave do objeto para `config.json`
- `ENCODER_S3_KEY` — chave do objeto para `label_encoder.pkl`
- Se usar boto3 fora do ambiente padrão, configurar `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` e `AWS_REGION` como variáveis de ambiente no Render.

Observações sobre armazenamento do modelo
- Recomendo usar S3 Standard. Bloqueie acesso público e use presigned URLs ou IAM com permissões mínimas (`s3:GetObject`).
- Habilite versioning e SSE-KMS para segurança/rollback.

Deploy no Render (passo-a-passo resumido)
1. Commit e push do repositório ao GitHub.
2. No Render, crie um novo Web Service apontando para o repositório.
3. Configure as `Environment` variables (ex.: presigned URLs ou S3 creds/keys acima).
4. Start command recomendado:

```bash
gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
```
(Alternativa) `uvicorn main:app --host 0.0.0.0 --port $PORT` — serve bem para testes, mas `gunicorn` é mais robusto em produção.

Git — comandos rápidos para subir ao GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin git@github.com:SEU_USUARIO/SEU_REPO.git
git branch -M main
git push -u origin main
```

`.gitignore`
- `venv/` ou `ENV/` conforme seu virtualenv
- `__pycache__/`
- `*.pyc`
- `.env`
- `*.h5` (ou inclua apenas os modelos se quiser subir)

# https://github.com/cristoferaraujokahala/sonor-health-api/releases/download/V1.0/config.json

# https://github.com/cristoferaraujokahala/sonor-health-api/releases/download/V1.0/label_encoder.pkl

# https://github.com/cristoferaraujokahala/sonor-health-api/releases/download/V1.0/model.weights.h5
