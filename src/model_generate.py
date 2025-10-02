import json
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# 設定値
MAX_SEQUENCE_LENGTH = 10
VOCAB_SIZE = 500

# スクリプトの場所を基準にパスを解決
BASE_DIR = Path(__file__).resolve().parent.parent

# データのロード
with open(BASE_DIR / 'data/qa_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
with open(BASE_DIR / 'data/dictionary.json', 'r', encoding='utf-8') as f: # 修正不要
    dictionary = json.load(f)

# 辞書置換関数
def apply_dictionary(text, dictionary):
    for key, value in dictionary.items():
        text = text.replace(key, value)
    return text

# 質問とラベルの抽出、辞書置換適用
questions = []
labels = []
for item in data:
    for q in item['questions']:
        # 意図認識精度向上のための辞書置換を適用
        processed_q = apply_dictionary(q, dictionary)
        questions.append(processed_q)
        labels.append(item['intent'])

# ラベルのエンコーディング (意図名 -> ID)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
Y = to_categorical(integer_encoded)
intent_classes = list(label_encoder.classes_) # JS側で使用

# トークナイザーの学習と保存
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer.fit_on_texts(questions)

# テキストをシーケンスに変換し、パディング
sequences = tokenizer.texts_to_sequences(questions)
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# 💡 重要なエクスポート (JS側で利用)
# 1. トークナイザーのボキャブラリをJSONとして保存
tokenizer_json = tokenizer.to_json(ensure_ascii=False)
with open(BASE_DIR / 'public/data/tokenizer_config.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)
# 2. 意図のクラス名を保存
with open(BASE_DIR / 'public/data/intent_classes.json', 'w', encoding='utf-8') as f:
    json.dump(intent_classes, f, ensure_ascii=False)
    

# モデルの構築 (シンプルなBOW/埋め込み平均モデル)
model = Sequential([
    Embedding(VOCAB_SIZE, 16, input_length=MAX_SEQUENCE_LENGTH),
    GlobalAveragePooling1D(), # 全単語の埋め込みベクトルの平均を取る
    Dense(16, activation='relu'),
    Dense(len(intent_classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# モデルの学習 (X, Yは上で作成したものを使用)
model.fit(X, Y, epochs=50, verbose=0)

# 出力先ディレクトリを作成
saved_model_path = BASE_DIR / "public/model/saved_model"
tfjs_model_path = BASE_DIR / "public/model"
saved_model_path.parent.mkdir(parents=True, exist_ok=True)

# モデルの保存 (SavedModel形式)
model.save(saved_model_path)

print(f"Model saved to {saved_model_path}")
print("\nTo convert the model to TensorFlow.js format, run the following command:")
print("-" * 70)
print(f"tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model \"{saved_model_path}\" \"{tfjs_model_path}\"")
print("-" * 70)

# SavedModelをtfjs形式に変換
# (コマンド例)
# tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model "c:\programs\GitHubWS\AI-ChatBot\public\model\saved_model" "c:\programs\GitHubWS\AI-ChatBot\public\model"