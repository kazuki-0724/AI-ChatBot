import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# テストページ：http://127.0.0.1:5000/

# --- AIモデルの準備 ---

# 1. 日本語対応の事前学習済みモデルをロード
#    初回実行時にモデルのダウンロードが自動的に行われます。
model = SentenceTransformer('cl-tohoku/bert-base-japanese-whole-word-masking')

# スクリプト自身の場所を基準に、ファイルの絶対パスを解決
BASE_DIR = Path(__file__).resolve().parent
FAQ_FILE_PATH = BASE_DIR.parent / "data" / "faq.json"

# 2. FAQデータを読み込む
with open(FAQ_FILE_PATH, 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

# 質問文のリストと、それに対応する回答のリストを作成
questions = []
answers = []
for item in faq_data:
    # 1つの回答に複数の質問が紐づいているため、それぞれを展開してリストに追加
    for q in item['questions']:
        questions.append(q)
        answers.append(item['answer'])

# 3. FAQの全質問文をベクトルに変換（エンコーディング）
question_vectors = model.encode(questions, convert_to_tensor=True)

# --- [確認用] ベクトルの次元数を出力 ---
# vector_dimension = question_vectors.shape[1]
# print(f"AIモデルが生成するベクトルの次元数: {vector_dimension}") # 768
# ------------------------------------

# --- Flaskアプリケーションのセットアップ ---

# publicディレクトリの絶対パスを指定
PUBLIC_DIR = BASE_DIR.parent / "public"
# Flaskアプリケーションのインスタンスを作成
app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='')

# CORS（クロスオリジンリソース共有）を有効にする
CORS(app)

# ルートURL ('/') にアクセスされたときにindex.htmlを返す
@app.route('/')
def index():
    # static_folderから'index.html'を返す
    return send_from_directory(app.static_folder, 'index.html')

# チャットのAPIエンドポイント
@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    print(f"ユーザーからのメッセージ: {user_message}")

    if not user_message:
        return jsonify({'reply': 'メッセージを入力してください。'})

    # --- AIによる回答生成 ---
    # 1. ユーザーの質問をベクトルに変換
    user_vector = model.encode(user_message, convert_to_tensor=True)

    # 2. FAQの各質問とユーザーの質問のコサイン類似度を計算
    similarities = util.cos_sim(user_vector, question_vectors)

    # 3. 最も類似度が高い質問のインデックスを取得
    most_similar_index = np.argmax(similarities.cpu().numpy()) # Tensorをnumpyに変換
    max_similarity = similarities[0][most_similar_index].item()

    # 4. 類似度が一定の閾値（ここでは0.6）より高い場合、対応する回答を返す
    if max_similarity > 0.6:
        bot_reply = answers[most_similar_index]
    else:
        bot_reply = "申し訳ありません、よく分かりませんでした。別の言葉で質問していただけますか？"

    return jsonify({'reply': bot_reply})

if __name__ == '__main__':
    # 開発用サーバーを起動
    app.run(debug=True, port=5000)