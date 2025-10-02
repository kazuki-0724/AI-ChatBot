let tokenizer = null;
let model = null;
let qaData = null;
let intentClasses = null;
let wordIndex = null;

const MAX_SEQUENCE_LENGTH = 10; // Python側で設定した値と合わせる
const OOV_TOKEN = "<unk>"; // Python側で設定した値と合わせる

/**
 * 必要なリソース（モデル、トークナイザー等）を非同期で読み込み、初期化する
 */
async function initializeBot() {
  try {
    const [loadedModel, loadedQaData, loadedIntents, tokenizerConfig] = await Promise.all([
      tf.loadGraphModel('model/model.json'),
      fetch('data/qa_data.json').then(res => res.json()),
      fetch('data/intent_classes.json').then(res => res.json()),
      fetch('data/tokenizer_config.json').then(res => res.json())
    ]);

    model = loadedModel;
    qaData = loadedQaData;
    intentClasses = loadedIntents;
    wordIndex = JSON.parse(tokenizerConfig.config.word_index);

    // kuromoji.jsの初期化
    tokenizer = await new Promise((resolve, reject) => {
      kuromoji.builder({ dicPath: "https://cdn.jsdelivr.net/npm/kuromoji@0.1.2/dict/" }).build((err, tkz) => {
        if (err) return reject(err);
        resolve(tkz);
      });
    });

    main(); // すべての読み込み完了後、main関数を実行
  } catch (err) {
    console.error("初期化エラー:", err);
    document.getElementById('loading-overlay').innerHTML = '<div>初期化に失敗しました。<br>コンソールを確認してください。</div>';
  }
}

function tokenize(text) {
  if (!tokenizer) return [];
  return tokenizer.tokenize(text)
    .map(token => token.basic_form === "*" ? token.surface_form : token.basic_form);
}

/**
 * テキストを単語IDのシーケンスに変換し、パディングする
 * @param {string} text 入力テキスト
 * @returns {number[]} パディング済みの単語IDシーケンス
 */
function textToSequence(text) {
  const tokens = tokenize(text);
  const sequence = tokens.map(word => {
    return wordIndex[word] || wordIndex[OOV_TOKEN];
  });
  // パディング
  const paddedSequence = new Array(MAX_SEQUENCE_LENGTH).fill(0);
  const len = Math.min(sequence.length, MAX_SEQUENCE_LENGTH);
  for (let i = 0; i < len; i++) {
    paddedSequence[i] = sequence[i];
  }
  return paddedSequence;
}

function main() {
  const chat = document.getElementById('chat');
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const loadingOverlay = document.getElementById('loading-overlay');

  function addChat(text, isUser) {
    const div = document.createElement('div');
    div.className = 'bubble ' + (isUser ? 'user' : 'bot');
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  // UIの有効化
  userInput.disabled = false;
  sendBtn.disabled = false;
  loadingOverlay.style.display = 'none';
  addChat("こんにちは！何か質問はありますか？", false);

  sendBtn.onclick = async () => {
    const text = userInput.value.trim();
    if (!text) return;
    addChat(text, true);
    userInput.value = '';
    sendBtn.disabled = true;

    // 入力テキストをモデルが受け取れる形式に変換
    const sequence = textToSequence(text);
    const vec = tf.tensor2d([sequence], [1, MAX_SEQUENCE_LENGTH], 'int32');

    // 推論実行
    const pred = model.predict(vec);
    const predData = await pred.data();
    const idx = predData.indexOf(Math.max(...predData));
    const confidence = predData[idx];
    const predictedIntent = intentClasses[idx];

    // 対応する回答を検索
    const response = qaData.find(item => item.intent === predictedIntent)?.answer || "申し訳ありませんが、よく分かりません。";

    addChat(`${response} (信頼度: ${confidence.toFixed(2)})`, false);

    vec.dispose();
    pred.dispose();
    sendBtn.disabled = false;
  };

  userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !sendBtn.disabled) {
      e.preventDefault(); // フォームの送信を防ぐ
      sendBtn.click();
    }
  });
}

// 初期化処理を開始
initializeBot();