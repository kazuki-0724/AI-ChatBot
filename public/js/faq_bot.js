let tokenizer = null;

// kuromoji.jsとFAQデータを非同期で読み込む
Promise.all([
  new Promise((resolve, reject) => {
    kuromoji.builder({ dicPath: "https://cdn.jsdelivr.net/npm/kuromoji@0.1.2/dict/" }).build(function (err, tkz) {
      if (err) return reject(err);
      tokenizer = tkz;
      resolve();
    });
  }),
  fetch('data/faqs.json').then(res => res.json())
]).then(([_, faqs]) => {
  main(faqs); // 読み込み完了後、FAQデータを渡してmain関数を実行
}).catch(err => {
  console.error("初期化エラー:", err);
  alert("チャットボットの初期化に失敗しました。");
});

function tokenize(text) {
  if (!tokenizer) return [];
  // 形態素解析して名詞・動詞・形容詞など主要な単語のみ抽出
  return tokenizer.tokenize(text)
    .filter(token => ["名詞", "動詞", "形容詞"].includes(token.pos))
    .map(token => token.basic_form === "*" ? token.surface_form : token.basic_form);
}

function main(faqs) {
  const chat = document.getElementById('chat');
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const loadingOverlay = document.getElementById('loading-overlay');

  function setUIEnabled(enabled) {
    userInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
    loadingOverlay.style.display = enabled ? 'none' : 'flex';
  }

  // 全FAQ質問文を形態素解析
  const allTexts = faqs.map(f => f.q);
  const allTokens = allTexts.map(tokenize);
  const vocab = Array.from(new Set(allTokens.flat()));
  
  // TF-IDFの計算
  const idf = vocab.map(w => Math.log(allTexts.length / (allTokens.filter(tokens => tokens.includes(w)).length + 1)));
  
  function textToVec(text) {
    const tokens = tokenize(text);
    // TF (Term Frequency)
    const tf = vocab.map(w => tokens.filter(t => t === w).length / (tokens.length || 1));
    // TF-IDF
    return tf.map((v, i) => v * idf[i]);
  }
  
  // // 以前のtextToVec (Bag of Words)
  // function textToVec(text) {
  //   const tokens = tokenize(text);
  //   return vocab.map(w => tokens.filter(t => t === w).length);
  // }

  let xs, ys;
  let model = null;

  async function prepareModel() {
    try {
      model = await tf.loadLayersModel('indexeddb://faq-model-kuromoji');
      console.log('モデルをIndexedDBからロードしました');
    } catch (e) {
      console.log('保存済みモデルがありません。新規学習します');
      model = tf.sequential();
      model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [vocab.length] }));
      model.add(tf.layers.dropout({ rate: 0.5 })); // ドロップアウト層を追加して過学習を抑制
      model.add(tf.layers.dense({ units: faqs.length, activation: 'softmax' }));
      model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
      await model.fit(xs, ys, { epochs: 500, verbose: 0 }); // エポック数を少し増やして学習
      await model.save('indexeddb://faq-model-kuromoji');
      console.log('モデルをIndexedDBに保存しました');
    }
  }

  tf.setBackend('cpu').then(async () => {
    console.log('TensorFlow.js backend set to CPU');
    xs = tf.tensor2d(allTexts.map(textToVec));
    ys = tf.tensor2d(faqs.map((_, i) => {
      const arr = new Array(faqs.length).fill(0);
      arr[i] = 1;
      return arr;
    }));
    // 非同期処理がUIの準備完了より先に終わるようにawaitを追加
    await prepareModel();
    // モデルの準備が完了してからUIを有効化
    setUIEnabled(true);
    addChat("こんにちは！何か質問はありますか？", false);
  });

  setUIEnabled(false);

  // チャットUI
  function addChat(text, isUser) {
    const div = document.createElement('div');
    div.className = 'bubble ' + (isUser ? 'user' : 'bot');
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  sendBtn.onclick = async () => {
    const text = userInput.value.trim();
    if (!text) return;
    addChat(text, true);
    userInput.value = '';
    sendBtn.disabled = true;
    while (!model) await new Promise(r => setTimeout(r, 100));

    const vec = tf.tensor2d([textToVec(text)]);
    const pred = model.predict(vec);
    const predData = await pred.data();
    const idx = predData.indexOf(Math.max(...predData));
    const confidence = predData[idx];

    addChat(`${faqs[idx].a}（${confidence.toFixed(2)}）`, false);
    vec.dispose();
    pred.dispose();
    sendBtn.disabled = false;
  };

  userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !sendBtn.disabled) sendBtn.click();
  });
}