# --- ライブラリのインポート ---
# FastAPI本体、CORSミドルウェア、モデル定義、OpenAIクライアント、各種ユーティリティを読み込む
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import json
from openai import OpenAI
import logging

# --- ログ設定 ---
# INFOレベル以上をコンソールに出力
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPIアプリケーションの初期化 ---
# タイトルを指定してAPIサーバを立ち上げる
app = FastAPI(title="Code Learning Feedback API")

# --- CORS設定 ---
# 全オリジンからのアクセスを許可（開発時用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAIクライアントの遅延初期化用変数 ---
# 環境変数からキーを取得し、初回アクセス時にクライアントを生成
client = None

def get_openai_client():
    """
    OpenAIクライアントを取得（初回は環境変数からAPIキーを読み込み、例外処理付きで初期化）
    2回目以降は既存インスタンスを返却
    """
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API Key not found")
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize OpenAI client")
    return client

# --- Pydanticモデル定義 ---
# リクエストとレスポンスのデータ構造を型安全に定義

class CodeEvaluationRequest(BaseModel):
    # クライアントから受け取るコード評価リクエスト
    code: str
    task_id: int
    task_description: str

class CodeFeedback(BaseModel):
    # AIが返すフィードバックの詳細構造
    overall_score: int
    level: int
    is_correct: bool
    strengths: List[str]
    improvements: List[str]
    hints: List[str]
    detailed_feedback: str
    next_steps: str

class EvaluationResponse(BaseModel):
    # 評価APIの最終レスポンス構造
    success: bool
    feedback: CodeFeedback
    error: Optional[str] = None

# --- ルートエンドポイント ---
# サービス稼働確認用のシンプルな応答
@app.get("/")
async def root():
    return {"message": "Code Learning Feedback API"}

# --- OpenAI接続テスト用エンドポイント ---
# "Hello"を送ってAPIが動作するか確かめる
@app.get("/api/test-openai")
async def test_openai():
    try:
        openai_client = get_openai_client()
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        return {
            "status": "success",
            "message": "OpenAI API connection successful",
            "response_preview": response.choices[0].message.content[:50]
        }
    except Exception as e:
        logger.error(f"OpenAI API test failed: {str(e)}")
        return {
            "status": "error", 
            "message": str(e),
            "error_type": type(e).__name__
        }

# --- ヘルスチェックエンドポイント ---
# ファイルの存在や環境変数、クライアント初期化状況を返却
@app.get("/api/health")
async def health_check():
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_file_path = os.path.join(current_dir, "tasks.json")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_status = "configured" if openai_api_key else "not_configured"
    return {
        "status": "ok",
        "current_directory": current_dir,
        "tasks_file_path": tasks_file_path,
        "tasks_file_exists": os.path.exists(tasks_file_path),
        "files_in_directory": os.listdir(current_dir) if os.path.exists(current_dir) else [],
        "openai_api_key_status": openai_status,
        "openai_client_initialized": client is not None
    }

# --- コード評価エンドポイント ---
# 学習者のCコードを受け取り、OpenAIで構造化フィードバックを生成して返却
@app.post("/api/evaluate-code", response_model=EvaluationResponse)
async def evaluate_code(request: CodeEvaluationRequest):
    try:
        logger.info(f"Evaluating code for task {request.task_id}")
        
        # プロンプトを構築
        system_prompt = """
あなたはプログラミング学習支援AIです。学習者のCコードを評価し、建設的なフィードバックを提供してください。

評価基準：
1. コードの正確性（課題要求を満たしているか）
2. コードの品質（可読性、効率性）
3. プログラミング技法の適切な使用
4. エラーハンドリング

レベル評価：
- レベル1: 基本的な構文エラーがある、課題を理解していない
- レベル2: 構文は正しいが、論理エラーがある
- レベル3: 基本的な要求は満たしているが、改善点がある
- レベル4: 良いコードで、軽微な改善点のみ
- レベル5: 優秀なコード、模範的な実装

フィードバックは建設的で、学習者のモチベーションを維持するよう心がけてください。
"""

        user_prompt = f"""
課題 {request.task_id}: {request.task_description}

学習者のコード:
```c
{request.code}
```

このコードを評価してください。
"""

        # OpenAI API呼び出し（Structured Outputs使用）
        openai_client = get_openai_client()
        
        # まず通常のAPI呼び出しでテスト
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "code_feedback",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "overall_score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "level": {"type": "integer", "minimum": 1, "maximum": 5},
                                "is_correct": {"type": "boolean"},
                                "strengths": {"type": "array", "items": {"type": "string"}},
                                "improvements": {"type": "array", "items": {"type": "string"}},
                                "hints": {"type": "array", "items": {"type": "string"}},
                                "detailed_feedback": {"type": "string"},
                                "next_steps": {"type": "string"}
                            },
                            "required": ["overall_score", "level", "is_correct", "strengths", "improvements", "hints", "detailed_feedback", "next_steps"],
                            "additionalProperties": False
                        }
                    }
                },
                temperature=0.7
            )
            
            # JSONレスポンスをパース
            import json
            feedback_json = json.loads(completion.choices[0].message.content)
            
            # Pydanticモデルに変換
            feedback = CodeFeedback(
                overall_score=feedback_json.get("overall_score", 1),
                level=feedback_json.get("level", 1),
                is_correct=feedback_json.get("is_correct", False),
                strengths=feedback_json.get("strengths", []),
                improvements=feedback_json.get("improvements", ["コードを見直してください"]),
                hints=feedback_json.get("hints", ["基本的な構文を確認してください"]),
                detailed_feedback=feedback_json.get("detailed_feedback", "コードの評価ができませんでした"),
                next_steps=feedback_json.get("next_steps", "もう一度挑戦してください")
            )
            
        except Exception as api_error:
            logger.error(f"OpenAI API error: {str(api_error)}")
            # フォールバック: シンプルなフィードバックを生成
            feedback = CodeFeedback(
                overall_score=2,
                level=2,
                is_correct=False,
                strengths=["コードの提出ができています"],
                improvements=["APIエラーのため詳細な評価ができませんでした"],
                hints=["コードの構文を確認してください", "課題の要求を再確認してください"],
                detailed_feedback=f"申し訳ございませんが、AI評価システムにエラーが発生しました。エラー詳細: {str(api_error)}",
                next_steps="コードを見直して再度お試しください"
            )
        
        logger.info(f"Generated feedback with level {feedback.level}")
        
        return EvaluationResponse(
            success=True,
            feedback=feedback
        )

    except Exception as e:
        logger.error(f"Error evaluating code: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # より詳細なエラー情報を含むフォールバック
        error_message = str(e)
        if "Beta" in error_message:
            error_message = "OpenAI APIのバージョン互換性エラーが発生しました"
        elif "API" in error_message:
            error_message = "OpenAI APIとの通信エラーが発生しました"
        elif "Key" in error_message:
            error_message = "OpenAI API Keyの設定に問題があります"
        
        return EvaluationResponse(
            success=False,
            feedback=CodeFeedback(
                overall_score=1,
                level=1,
                is_correct=False,
                strengths=["コードの提出ができています"],
                improvements=[f"システムエラー: {error_message}"],
                hints=["しばらく待ってから再試行してください", "管理者に問題を報告してください"],
                detailed_feedback=f"申し訳ございませんが、コード評価システムにエラーが発生しました。技術的詳細: {error_message}",
                next_steps="システム復旧後に再度お試しください"
            ),
            error=str(e)
        )

# --- 課題一覧取得エンドポイント ---
# backend/tasks.json から全課題を読み込んで返す
@app.get("/api/tasks")
async def get_tasks():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tasks_file_path = os.path.join(current_dir, "tasks.json")
        with open(tasks_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Tasks file not found at {tasks_file_path}")
        raise HTTPException(status_code=404, detail="Tasks file not found")
    except Exception as e:
        logger.error(f"Error loading tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 特定課題取得エンドポイント ---
# 指定IDの課題を tasks.json から検索して返却
@app.get("/api/tasks/{task_id}")
async def get_task(task_id: int):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tasks_file_path = os.path.join(current_dir, "tasks.json")
        with open(tasks_file_path, "r", encoding="utf-8") as f:
            tasks_data = json.load(f)
        task = next((t for t in tasks_data["tasks"] if t["id"] == task_id), None)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except FileNotFoundError:
        logger.error(f"Tasks file not found at {tasks_file_path}")
        raise HTTPException(status_code=404, detail="Tasks file not found")
    except Exception as e:
        logger.error(f"Error loading task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- アプリ起動処理 ---
# 単体実行時は uvicorn でサーバを起動
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)