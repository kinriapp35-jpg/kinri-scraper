"""
銀行金利スクレイピング + LLM抽出スクリプト
GitHub Actionsから1日1回実行される
"""
import asyncio
import json
import os
import time
from pathlib import Path

import html2text
import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# ===== 設定 =====
CONFIG_PATH = Path(__file__).parent / "bank_config.json"
OUTPUT_PATH = Path(__file__).parent / "extracted_rates.json"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CF_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
WORKER_URL = os.environ.get("WORKER_URL", "")

# GitHub Models endpoint (GPT-4o-mini via GitHub)
GITHUB_MODELS_URL = "https://models.inference.ai.azure.com/chat/completions"
GITHUB_MODEL_NAME = "gpt-4o-mini"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


async def fetch_page_html(url: str, bank_id: str) -> str:
    """Playwrightでページを取得し、HTMLを返す"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)  # JS完全読込待ち
            html = await page.content()
            print(f"  [OK] {bank_id}: {len(html)} chars")
        except Exception as e:
            print(f"  [ERROR] {bank_id}: {e}")
            html = ""
        finally:
            await browser.close()

    return html


def html_to_markdown(html: str, bank_name: str) -> str:
    """HTML → Markdown変換（金利関連部分を抽出）"""
    soup = BeautifulSoup(html, "html.parser")

    # 不要な要素を除去
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # テーブルと金利関連テキストを優先抽出
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0  # 折り返しなし

    markdown = converter.handle(str(soup))

    # 長すぎる場合は金利関連部分のみ切り出し
    if len(markdown) > 8000:
        lines = markdown.split("\n")
        relevant_lines = []
        keywords = ["金利", "利率", "%", "年率", "普通預金", "利息", "優遇"]
        for i, line in enumerate(lines):
            if any(kw in line for kw in keywords):
                # 前後5行を含める
                start = max(0, i - 5)
                end = min(len(lines), i + 10)
                relevant_lines.extend(lines[start:end])
        markdown = "\n".join(dict.fromkeys(relevant_lines))  # 重複除去

    # 最大4000文字に制限（LLMのトークン節約）
    return markdown[:4000]


def build_llm_prompt(markdown: str, bank_config: dict) -> str:
    """LLMに送るプロンプトを構築"""
    terms_info = json.dumps(bank_config["terms"], ensure_ascii=False, indent=2)

    return f"""以下は「{bank_config['name']}」の金利ページから抽出したMarkdownテキストです。

この銀行の**普通預金金利**に関する情報を全て抽出してください。

## この銀行特有の用語辞書:
{terms_info}

## 抽出ルール:
1. 普通預金（または普通預金相当の商品）の金利のみ抽出
2. 定期預金は対象外
3. 条件付きの金利は「条件」を明記
4. 金利は年率(%)で統一
5. 複数の金利体系がある場合は全て抽出

## 出力形式（JSON）:
```json
{{
  "bank_id": "{bank_config['id']}",
  "bank_name": "{bank_config['name']}",
  "rates": [
    {{
      "type": "通常" or "優遇",
      "rate_percent": 0.30,
      "product_name": "普通預金",
      "condition": null or "条件の説明",
      "condition_name": null or "マネーブリッジ等の固有名称",
      "limit_amount": null or "300万円以下" 等の残高条件,
      "note": "補足情報"
    }}
  ],
  "updated_date": "ページに記載の更新日（あれば）"
}}
```

## ページのMarkdownテキスト:
{markdown}

JSONのみを出力してください。"""


async def extract_with_llm(prompt: str) -> dict:
    """GitHub Models (GPT-4o-mini) でJSON抽出"""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GITHUB_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "あなたは銀行金利データの抽出アシスタントです。与えられたテキストから正確に金利情報を抽出し、指定のJSON形式で出力してください。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(GITHUB_MODELS_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            print(f"  [LLM ERROR] {resp.status_code}: {resp.text[:200]}")
            return {}

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # JSONブロックを抽出
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            print(f"  [JSON PARSE ERROR] {e}")
            print(f"  Content: {content[:200]}")
            return {}


async def update_worker(all_rates: list):
    """Cloudflare Workerにデータを送信"""
    if not WORKER_URL:
        print("[SKIP] WORKER_URL not set, saving locally only")
        return

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{WORKER_URL}/api/admin/update-rates",
            json={"rates": all_rates},
            headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"}
        )
        if resp.status_code == 200:
            print(f"[OK] Worker updated with {len(all_rates)} banks")
        else:
            print(f"[ERROR] Worker update failed: {resp.status_code}")


async def main():
    config = load_config()
    banks = config["banks"]
    interval = config["scraping_notes"]["request_interval_seconds"]

    print(f"=== 金利スクレイピング開始: {len(banks)}行 ===")

    all_rates = []

    for bank in banks:
        print(f"\n--- {bank['name']} ({bank['id']}) ---")

        # 1. Playwrightでページ取得
        html = await fetch_page_html(bank["url"], bank["id"])
        if not html:
            continue

        # 2. HTML → Markdown変換
        markdown = html_to_markdown(html, bank["name"])
        print(f"  Markdown: {len(markdown)} chars")

        if len(markdown) < 50:
            print(f"  [SKIP] コンテンツが少なすぎる")
            continue

        # 3. LLMで金利データ抽出
        prompt = build_llm_prompt(markdown, bank)
        result = await extract_with_llm(prompt)

        if result and "rates" in result:
            all_rates.append(result)
            print(f"  [EXTRACTED] {len(result['rates'])} rate(s)")
        else:
            print(f"  [NO DATA]")

        # レート制限対策
        time.sleep(interval)

    # 4. 結果を保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_rates, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完了: {len(all_rates)}/{len(banks)}行のデータ抽出 ===")
    print(f"Output: {OUTPUT_PATH}")

    # 5. Cloudflare Workerに送信
    await update_worker(all_rates)


if __name__ == "__main__":
    asyncio.run(main())
