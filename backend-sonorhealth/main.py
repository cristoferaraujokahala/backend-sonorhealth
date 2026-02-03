from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
# reduzir logs informativos do TensorFlow (deve ser antes do import do TF)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import librosa
import numpy as np
import pickle
import json
import io  # Para ler o arquivo em memória

app = FastAPI(
    title="Classificador de Sons Respiratórios",
    description="API para classificar tosse/sons respiratórios usando modelo ICBHI treinado.",
    version="1.0"
)

# Habilita CORS para permitir que o app móvel (Expo) envie requisições
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção especifique as origens necessárias
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações (ajusta para o teu modelo)
MODEL_DIR = r'C:\API_venv\api\classificador_respiratorio'
MODEL_WEIGHTS_PATH = rf'{MODEL_DIR}\model.weights.h5'
MODEL_CONFIG_PATH = rf'{MODEL_DIR}\config.json'
ENCODER_PATH = r'C:\API_venv\api\label_encoder.pkl'
SR = 16000  # Taxa de amostragem
SEGMENT_SEC = 3.0  # Duração de cada segmento
N_MFCC = 40
MAX_LEN = 130

# Carregar modelo e encoder ao iniciar a API
print("Carregando modelo e encoder...")

# Carregar configuração do modelo
with open(MODEL_CONFIG_PATH, 'r') as f:
    full_config = json.load(f)

# Extrair apenas a configuração interna (ignora o wrapper)
model_config = full_config.get('config', full_config)

# Reconstruir o modelo a partir da configuração
model = tf.keras.Sequential.from_config(model_config)
model.load_weights(MODEL_WEIGHTS_PATH)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

# Função de extração de features (idêntica ao teu treino)
def extract_features(audio, sr=SR, n_mfcc=N_MFCC, max_len=MAX_LEN):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        rms = librosa.feature.rms(y=audio)[0]

        features = np.concatenate([
            mfcc, delta, delta2,
            chroma[:12], contrast[:7],
            zcr.reshape(1, -1), rms.reshape(1, -1)
        ], axis=0)

        if features.shape[0] > n_mfcc:
            features = features[:n_mfcc]
        else:
            features = np.pad(features, ((0, n_mfcc - features.shape[0]), (0, 0)), 'constant')

        if features.shape[1] > max_len:
            features = features[:, :max_len]
        else:
            features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), 'constant')

        return np.expand_dims(features, axis=-1)
    except Exception as e:
        raise ValueError(f"Erro ao extrair features: {e}")

# Endpoint principal para previsão
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Ler o áudio em memória (suporta .wav, .mp3, etc.)
        contents = await file.read()
        y, sr = librosa.load(io.BytesIO(contents), sr=None)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)

        duration = len(y) / SR
        n_segments = max(1, int(duration / SEGMENT_SEC) + (1 if duration % SEGMENT_SEC > 0.5 else 0))

        predictions = []
        confidences = []

        for i in range(n_segments):
            start = int(i * SEGMENT_SEC * SR)
            end = min(start + int(SEGMENT_SEC * SR), len(y))
            segment = y[start:end]

            if len(segment) < SR * 0.8:
                continue

            feat = extract_features(segment)
            feat = np.expand_dims(feat, axis=0)  # Batch dim

            pred_prob = model.predict(feat)[0]
            class_idx = np.argmax(pred_prob)
            class_name = le.inverse_transform([class_idx])[0]
            confidence = pred_prob[class_idx]

            predictions.append(class_name)
            confidences.append(float(confidence))  # Converter para float simples

        if not predictions:
            raise HTTPException(status_code=400, detail="Áudio muito curto ou inválido.")

        # Voto majoritário
        from collections import Counter
        most_common = Counter(predictions).most_common(1)
        final_class = most_common[0][0]
        avg_conf = np.mean([c for p, c in zip(predictions, confidences) if p == final_class])

        return JSONResponse({
            "classe_prevista": final_class,
            "confianca": f"{avg_conf:.2%}",
            "detalhes_segmentos": [{"classe": p, "confianca": f"{c:.2%}"} for p, c in zip(predictions, confidences)]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Rodar a API localmente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)