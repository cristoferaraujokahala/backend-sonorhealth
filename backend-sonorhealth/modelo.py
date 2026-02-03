
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import glob
from tqdm import tqdm
import kagglehub
import random
from collections import Counter
import pickle

warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("BAIXANDO DATASET ICBHI 2017 via kagglehub")
print("="*70)

try:
    print("Baixando Respiratory Sound Database...")
    dataset_path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
    print(f"Dataset em: {dataset_path}")
except Exception as e:
    print(f"Erro ao baixar: {e}")
    dataset_path = None

def find_dataset_files(base_path):
    if not base_path: return [], [], None
    wav_files = sorted(glob.glob(f"{base_path}/**/*.wav", recursive=True))
    txt_files = sorted(glob.glob(f"{base_path}/**/*.txt", recursive=True))
    diagnosis_csv = None
    for root, _, files in os.walk(base_path):
        for f in files:
            if 'patient_diagnosis' in f.lower() and f.endswith('.csv'):
                diagnosis_csv = os.path.join(root, f)
                break
        if diagnosis_csv: break
    print(f"🎵 WAV: {len(wav_files)} | TXT: {len(txt_files)}")
    if diagnosis_csv:
        print(f"📋 Diagnósticos: {os.path.basename(diagnosis_csv)}")
    return wav_files, txt_files, diagnosis_csv

wav_files, _, diagnosis_file = find_dataset_files(dataset_path)

def load_diagnosis_mapping(diag_path):
    if not diag_path: return {}
    try:
        df = pd.read_csv(diag_path, header=None, names=['Patient ID', 'Diagnosis'])
        diagnosis_dict = {}
        for _, row in df.iterrows():
            pid = str(row['Patient ID']).strip()
            diag = str(row['Diagnosis']).strip().lower()
            diagnosis_dict[pid] = diag
        print(f"✅ {len(diagnosis_dict)} diagnósticos carregados")
        print("Distribuição:", pd.Series(diagnosis_dict.values()).value_counts())
        return diagnosis_dict
    except Exception as e:
        print(f"Erro ao ler diagnósticos: {e}")
        return {}

diagnosis_dict = load_diagnosis_mapping(diagnosis_file)

def get_label_and_patient_id(filename, diag_dict):
    basename = os.path.basename(filename).replace('.wav', '')
    parts = basename.split('_')
    if len(parts) < 3: return None, None
    patient_id = parts[0].strip()
    if patient_id not in diag_dict: return None, patient_id
    diag = diag_dict[patient_id].lower()

    if 'healthy' in diag:
        return 'normal', patient_id
    elif 'pneumonia' in diag:
        return 'pneumonia', patient_id
    elif 'bronchiectasis' in diag or 'bronchiolitis' in diag or 'bronchi' in diag:
        return 'bronquite', patient_id
    else:
        return None, None

def extract_features(audio, sr=16000, n_mfcc=40, max_len=130):
    try:
        mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        rms = librosa.feature.rms(y=audio)[0]

        features = np.concatenate([
            mfcc, delta, delta2,
            chroma[:12], contrast[:7],
            zcr.reshape(1,-1), rms.reshape(1,-1)
        ], axis=0)

        if features.shape[0] > n_mfcc:
            features = features[:n_mfcc]
        else:
            features = np.pad(features, ((0, n_mfcc - features.shape[0]), (0,0)), mode='constant')

        if features.shape[1] > max_len:
            features = features[:, :max_len]
        else:
            features = np.pad(features, ((0,0), (0, max_len - features.shape[1])), mode='constant')

        return np.expand_dims(features, axis=-1)
    except:
        return None

def process_dataset(wav_list, diag_dict, max_files=300, segment_sec=3.0):
    X, y, patients = [], [], []
    selected = random.sample(wav_list, min(max_files, len(wav_list))) if max_files else wav_list

    for filepath in tqdm(selected, desc="Processando"):
        label, pid = get_label_and_patient_id(filepath, diag_dict)
        if label is None or label == 'outro': continue

        try:
            y_audio, sr = librosa.load(filepath, sr=None)
            if sr != 16000:
                y_audio = librosa.resample(y_audio, orig_sr=sr, target_sr=16000)
            sr = 16000

            duration = len(y_audio) / sr
            n_segments = max(1, int(duration / segment_sec) + (1 if duration % segment_sec > 0.5 else 0))

            for i in range(n_segments):
                start = int(i * segment_sec * sr)
                end = min(start + int(segment_sec * sr), len(y_audio))
                segment = y_audio[start:end]
                if len(segment) < sr * 0.8: continue

                feat = extract_features(segment, sr)
                if feat is not None:
                    X.append(feat)
                    y.append(label)
                    patients.append(pid)
        except Exception as e:
            print(f"Erro {os.path.basename(filepath)}: {e}")
            continue

    if not X: return None, None, None
    X = np.array(X)
    y = np.array(y)
    patients = np.array(patients)
    print(f"\n→ {len(X)} segmentos | Classes: {Counter(y)}")
    return X, y, patients

print("\n" + "="*70)
print("PROCESSANDO DATASET")
print("="*70)

X, y, patient_ids = process_dataset(wav_files, diagnosis_dict, max_files=300, segment_sec=3.0)

if X is None or len(X) == 0:
    print("Falha no processamento.")
else:
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("\nClasses:")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls:12} → {i} ({np.sum(y_encoded==i)})")

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights_dict = dict(enumerate(class_weights))

# Divisão por paciente
unique_pats = np.unique(patient_ids)
train_pats, val_pats = train_test_split(unique_pats, test_size=0.20, random_state=42)
train_mask = np.isin(patient_ids, train_pats)
val_mask   = np.isin(patient_ids, val_pats)

X_train, y_train = X[train_mask], y_encoded[train_mask]
X_val,   y_val   = X[val_mask],   y_encoded[val_mask]

print(f"Treino: {len(X_train)} seg | Val: {len(X_val)} seg")

# MODELO
def build_model(input_shape, num_classes):
  model = keras.Sequential([
      layers.Input(shape=input_shape),
      layers.Conv2D(32, (3,3), activation='relu', padding='same'),
      layers.BatchNormalization(),
      layers.MaxPooling2D((2,2)),
      layers.Dropout(0.25),

      layers.Conv2D(64, (3,3), activation='relu', padding='same'),
      layers.BatchNormalization(),
      layers.MaxPooling2D((2,2)),
      layers.Dropout(0.3),

      layers.Conv2D(128, (3,3), activation='relu', padding='same'),
      layers.BatchNormalization(),
      layers.GlobalAveragePooling2D(),
      layers.Dropout(0.4),

      layers.Dense(128, activation='relu'),
      layers.BatchNormalization(),
      layers.Dropout(0.3),
      layers.Dense(num_classes, activation='softmax')
  ])
  return model

model = build_model(X_train.shape[1:], len(le.classes_))

model.compile(
  optimizer=keras.optimizers.Adam(learning_rate=0.001),
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']   # REMOVIDA AUC para evitar o erro
)

callbacks = [
  keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
  keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
  keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\nTreinando...")
history = model.fit(
  X_train, y_train,
  validation_data=(X_val, y_val),
  batch_size=32,
  epochs=50,
  class_weight=class_weights_dict,
  callbacks=callbacks,
  verbose=1
)

print("\n" + "="*70)
print("AVALIAÇÃO FINAL DO MODELO ESTRATIFICADO COM CLASSES ATUALIZADAS")
print("="*70)

# Avalie o modelo usando o conjunto de validação estratificado
loss_re_stratified, acc_re_stratified = model.evaluate(X_val, y_val, verbose=0)
print(f"Acurácia (estratificada, classes atualizadas): {acc_re_stratified:.4f}")

# Gere previsões de probabilidade (`y_pred_prob_re_stratified`) para `X_val_stratified`
y_pred_prob_re_stratified = model.predict(X_val, verbose=0)

# Converta as probabilidades em rótulos de classe (`y_pred_re_stratified`)
y_pred_re_stratified = np.argmax(y_pred_prob_re_stratified, axis=1)

print("\nClassification Report (estratificado, classes atualizadas):")
# Imprima o relatório de classificação
unique_val_labels_re_stratified = np.unique(y_val)
filtered_target_names_re_stratified = [le.classes_[label] for label in unique_val_labels_re_stratified]
print(classification_report(y_val, y_pred_re_stratified, labels=unique_val_labels_re_stratified, target_names=filtered_target_names_re_stratified, digits=4))

# Calcule as pontuações AUC (one-vs-rest) para cada classe
auc_scores_re_stratified = {}
for i, cls in enumerate(le.classes_):
    if i in unique_val_labels_re_stratified:
        try:
            auc = roc_auc_score(y_val == i, y_pred_prob_re_stratified[:, i])
            auc_scores_re_stratified[cls] = auc
        except ValueError:
            auc_scores_re_stratified[cls] = np.nan
    else:
        auc_scores_re_stratified[cls] = np.nan
print("\nAUC por classe (OvR, estratificado, classes atualizadas):")
for cls, score in auc_scores_re_stratified.items():
    print(f"  {cls:12}: {score:.4f}")

# Gere e exiba uma matriz de confusão
cm_re_stratified = confusion_matrix(y_val, y_pred_re_stratified, labels=unique_val_labels_re_stratified)
plt.figure(figsize=(8,6))
sns.heatmap(cm_re_stratified, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_target_names_re_stratified, yticklabels=filtered_target_names_re_stratified)
plt.title('Matriz de Confusão (Dados Estratificados, Classes Atualizadas)')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

model.save('classificador_respiratorio.h5')
with open('label_encoder_stratified_updated_classes.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\nModelo estratificado (classes atualizadas) salvo com sucesso.")
