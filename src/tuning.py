import json
from sentence_transformers import InputExample, SentenceTransformer, losses, util
import random
from torch.utils.data import DataLoader
from pathlib import Path

print("モデルのロードを開始します")
model = SentenceTransformer('cl-tohoku/bert-base-japanese-whole-word-masking')

# --- データ準備フェーズ ---

# 1. FAQデータのロード（前回のコードから再利用）
FAQ_FILE_PATH = Path(__file__).resolve().parent.parent / "data" / "faq.json"
with open(FAQ_FILE_PATH, 'r', encoding='utf-8') as f:
    faq_data = json.load(f)

# 2. 学習サンプル（InputExample）の作成
triplet_train_examples = []
mnr_train_examples = []
in_domain_answers = []
in_domain_questions = []
out_of_domain_questions = []

# FAQデータを「範囲内」と「範囲外」に分離する
for item in faq_data:
    if "分かりませんでした" in item['answer']:
        out_of_domain_questions.extend(item['questions'])
    else:
        in_domain_answers.append(item['answer'])
        in_domain_questions.extend(item['questions'])

print("学習サンプルの作成を開始します...")
# --- タスク1: TripletLoss用の学習データ作成 ---
# FAQ内のトピックを正確に区別する能力を学習
for item in faq_data:
    if "分かりませんでした" in item['answer']:
        continue # 範囲外の質問はアンカーとして使用しない

    positive_answer = item['answer']
    for question in item['questions']:
        # ハードネガティブサンプリング
        other_answers = [ans for ans in in_domain_answers if ans != positive_answer]
        negative_candidates = random.sample(other_answers, min(5, len(other_answers)))
        question_embedding = model.encode(question, convert_to_tensor=True)
        candidate_embeddings = model.encode(negative_candidates, convert_to_tensor=True)
        similarities = util.cos_sim(question_embedding, candidate_embeddings)
        hard_negative_answer = negative_candidates[similarities.argmax()]
        triplet_train_examples.append(InputExample(texts=[question, positive_answer, hard_negative_answer]))

# --- タスク2: MultipleNegativesRankingLoss用の学習データ作成 ---
# FAQ範囲内の質問と範囲外の質問を区別する能力を学習
for q in in_domain_questions:
    mnr_train_examples.append(InputExample(texts=[q, random.choice(out_of_domain_questions)]))

# 3. DataLoaderの作成
triplet_dataloader = DataLoader(triplet_train_examples, shuffle=True, batch_size=16)
mnr_dataloader = DataLoader(mnr_train_examples, shuffle=True, batch_size=16)

# --- ファインチューニングフェーズ ---
print("モデルのファインチューニングを開始します...")

# 2. 複数の損失関数を定義
triplet_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.5)
mnr_loss = losses.MultipleNegativesRankingLoss(model=model)

# ファインチューニング後のモデルを保存するパスを定義
OUTPUT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fine-tuned-faq-bot"
OUTPUT_MODEL_PATH.mkdir(parents=True, exist_ok=True) # 保存先ディレクトリがなければ作成

# 3. モデルの学習（ファインチューニング）の実行
# num_epochs: 学習回数。小さいFAQデータであれば1〜5エポック程度で十分なことが多い
model.fit(
    train_objectives=[
        (triplet_dataloader, triplet_loss),
        (mnr_dataloader, mnr_loss)
    ],
    epochs=3, # 例として3エポック
    warmup_steps=100, # 学習率を徐々に上げるためのステップ数
    output_path=str(OUTPUT_MODEL_PATH), # ファインチューニング後のモデルの保存先
    show_progress_bar=True 
)

# 4. モデルの保存
# 学習が完了すると、'path_to_fine_tuned_model'に新しいモデルが保存されます
print(f"ファインチューニングが完了し、モデルを {OUTPUT_MODEL_PATH} に保存しました。")