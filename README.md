# Code Learning Feedback System

C言語学習者向けのAIフィードバックシステムです。学習者が書いたコードと選択した課題内容を元に、OpenAI APIのStructured Outputsを使用して詳細なフィードバックを提供します。

## 🚀 特徴

- **111個の課題**: プログラミングIの授業の課題
- **AIフィードバック**: OpenAI GPT-4を使用した詳細なコード評価
- **Structured Outputs**: 一貫性のある構造化されたフィードバック
- **リアルタイム実行**: Wandbox APIを使用したC言語コードの実行
- **Docker環境**: 簡単なセットアップと運用

## 📁 ファイル構成

```
project/
├── docker-compose.yml          # Docker Compose設定
├── nginx.conf                  # Nginx設定
├── .env.template              # 環境変数テンプレート
├── README.md                  # このファイル
├── frontend/                  # フロントエンド
│   ├── index.html            # メインHTML
│   └── styles.css            # CSS
└── backend/                   # バックエンド
    ├── Dockerfile            # Python Dockerfile
    ├── requirements.txt      # Python依存関係
    ├── main.py              # FastAPI アプリケーション
    └── tasks.json           # 課題データ
```

## 🛠️ セットアップ

### 1. 環境変数の設定

```bash
# .env.templateをコピーして.envファイルを作成
cp .env.template .env

# .envファイルを編集してOpenAI API Keyを設定
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. Docker起動

```bash
# システム起動
docker-compose up -d

# ログ確認
docker-compose logs -f
```

### 4. アクセス

- **フロントエンド**: http://localhost:8080
- **API**: http://localhost:8000
- **API ドキュメント**: http://localhost:8000/docs

## 🎯 使用方法

1. **課題選択**: プルダウンまたは課題番号入力で課題を選択
2. **コード作成**: CodeMirrorエディタでC言語コードを記述
3. **コード実行**: 「コード実行」ボタンでWandbox APIによるコンパイル・実行
4. **AI評価**: 「🤖 AI Help - コード評価」ボタンでAIフィードバックを取得
5. **結果確認**: 実行結果とAIフィードバックを個別に確認

## 📊 ボタンの使い分け

- **「コード実行」**: Wandbox APIでC言語コードをコンパイル・実行（実行結果のみ）
- **「🤖 AI Help - コード評価」**: OpenAI APIでコードを分析し詳細なフィードバックを提供

## 🤖 AIフィードバックの内容

AIは以下の項目でコードを評価します：

- **総合スコア**: 1-5点での評価
- **レベル**: 学習レベル（1-5）
- **正解判定**: 課題要件を満たしているか
- **良い点**: コードの優れた部分
- **改善点**: 修正すべき箇所
- **ヒント**: 学習のための具体的なアドバイス
- **詳細フィードバック**: 総合的な評価コメント
- **次のステップ**: 今後の学習方針

## 🔧 開発

### バックエンド開発

```bash
# バックエンドのみ再起動
docker-compose restart backend

# バックエンドのログ確認
docker-compose logs backend
```

## 📋 API仕様

### POST /api/evaluate-code
コード評価用エンドポイント

**リクエスト:**
```json
{
  "code": "C言語のコード",
  "task_id": 1,
  "task_description": "課題の説明"
}
```

**レスポンス:**
```json
{
  "success": true,
  "feedback": {
    "overall_score": 4,
    "level": 3,
    "is_correct": true,
    "strengths": ["良い点のリスト"],
    "improvements": ["改善点のリスト"],
    "hints": ["ヒントのリスト"],
    "detailed_feedback": "詳細なフィードバック",
    "next_steps": "次のステップ"
  }
}
```

### GET /api/tasks
全課題取得

### GET /api/tasks/{task_id}
特定課題取得

## 🚨 トラブルシューティング

### よくある問題

1. **OpenAI API エラー**
   - API Keyが正しく設定されているか確認
   - API利用制限に達していないか確認

2. **Docker起動エラー**
   - ポート8080, 8000が使用されていないか確認
   - Docker Desktopが起動しているか確認

3. **フロントエンドからAPIにアクセスできない**
   - `docker-compose logs nginx`でNginxログを確認
   - CORSエラーの場合はバックエンドのCORS設定を確認

## 🔄 停止・クリーンアップ

```bash
# システム停止
docker-compose down

# 完全クリーンアップ（ボリューム削除）
docker-compose down -v

# イメージも削除
docker-compose down --rmi all
```
