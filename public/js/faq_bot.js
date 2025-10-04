document.addEventListener('DOMContentLoaded', () => {
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const chat = document.getElementById('chat');
  const loadingOverlay = document.getElementById('loading-overlay');

  // ダミーの準備時間（将来的にはモデルの読み込みなど）
  setTimeout(() => {
    loadingOverlay.style.display = 'none';
  }, 1000);

  // メッセージをチャットに追加する関数
  const addMessage = (text, sender) => {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);
    messageElement.textContent = text;
    chat.appendChild(messageElement);
    // 最新のメッセージが見えるようにスクロール
    chat.scrollTop = chat.scrollHeight;
  };

  // ボットからの返信を処理する関数
  const getBotResponse = async (userMessage) => {
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!response.ok) {
        throw new Error('サーバーからの応答がありません。');
      }

      const data = await response.json();
      addMessage(data.reply, 'bot');

    } catch (error) {
      console.error('エラー:', error);
      addMessage('申し訳ありません、エラーが発生しました。', 'bot');
    }
  };

  // メッセージを送信する関数
  const sendMessage = () => {
    const message = userInput.value.trim();
    if (message) {
      addMessage(message, 'user');
      userInput.value = '';
      sendBtn.disabled = true; // 連続送信を防ぐためにボタンを無効化

      // ボットが少し考えているように見せる
      setTimeout(() => {
        getBotResponse(message);
        sendBtn.disabled = false; // 応答が来たらボタンを有効化
      }, 500);
    }
  };

  // 送信ボタンのクリックイベント
  sendBtn.addEventListener('click', sendMessage);

  // Enterキーでの送信
  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  });

  // 初期メッセージ
  setTimeout(() => {
    addMessage('こんにちは！何か質問はありますか？', 'bot');
  }, 1200);
});