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

CONFIG_PATH = Path(__file__).parent / "bank_config.json"
OUTPUT_PATH = Path(__file__).parent / "extracted_rates.json"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
WORKER_URL = os.environ.get("WORKER_URL", "")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "")

GITHUB_MODELS_URL = "https://models.inference.ai.azure.com/chat/completions"
GITHUB_MODEL_NAME = "gpt-4o-mini"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


async def fetch_page_html(url: str, bank_id: str) -> str:
    """Playwrightでページを取得"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            locale="ja-JP",
        )
        page = await context.new_page()

        try:
            # domcontentloadedで待つ(networkidleだとタイムアウトしやすい)
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(3000)
            html = await page.content()
            print(f"  [OK] {bank_id}: {len(html)} chars")
        except Exception as e:
            print(f"  [ERROR] {bank_id}: {e}")
            html = ""
        finally:
            await browser.close()

    return html


def html_to_markdown(html: str) -> str:
    """HTML -> Markdown変換"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0

    markdown = converter.handle(str(soup))

    # 金利関連部分を優先抽出
    if len(markdown) > 6000:
        lines = markdown.split("\n")
        relevant = []
        keywords = ["金利", "利率", "%", "年率", "普通預金", "利息", "優遇", "ステージ", "連携"]
        for i, line in enumerate(lines):
            if any(kw in line for kw in keywords):
                start = max(0, i - 3)
                end = min(len(lines), i + 8)
                relevant.extend(lines[start:end])
        markdown = "\n".join(dict.fromkeys(relevant))

    return markdown[:4000]


def build_llm_prompt(markdown: str, bank_config: dict) -> str:
    terms_info = json.dumps(bank_config["terms"], ensure_ascii=False, indent=2)
    return f"""以下は「{bank_config['name']}」の金利ページから抽出したテキストです。

この銀行の**普通預金金利**を全て抽出してください。

## 用語辞書:
{terms_info}

## ルール:
1. 普通預金（相当商品含む）の金利のみ
2. 定期預金は対象外
3. 条件付きは「条件」を明記
4. 年率(%)で統一

## 出力JSON:
```json
{{
  "bank_id": "{bank_config['id']}",
  "bank_name": "{bank_config['name']}",
  "category": "{bank_config['category']}",
  "rates": [
    {{
      "type": "通常 or 優遇",
      "rate_percent": 0.30,
      "product_name": "普通預金",
      "condition": null or "条件説明",
      "condition_name": null or "固有名称",
      "limit_amount": null or "残高条件",
      "note": "補足"
    }}
  ]
}}
```

## テキスト:
{markdown}

JSONのみ出力:"""


async def extract_with_llm(prompt: str) -> dict:
    if not GITHUB_TOKEN:
        print("  [SKIP] No GITHUB_TOKEN")
        return {}

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GITHUB_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "銀行金利データの抽出アシスタント。正確にJSON出力。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(GITHUB_MODELS_URL, json=payload, headers=headers)
            if resp.status_code != 200:
                print(f"  [LLM ERROR] {resp.status_code}: {resp.text[:100]}")
                return {}

            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except Exception as e:
            print(f"  [LLM EXCEPTION] {e}")
            return {}


async def update_worker(all_rates: list):
    if not WORKER_URL or not CF_API_TOKEN:
        print("[SKIP] Worker update - no URL/token")
        return

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{WORKER_URL}/api/admin/update-rates",
                json={"rates": all_rates},
                headers={"Authorization": f"Bearer {CF_API_TOKEN}"}
            )
            print(f"[Worker] {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"[Worker ERROR] {e}")


async def main():
    config = load_config()
    banks = config["banks"]
    interval = config["scraping_notes"]["request_interval_seconds"]

    print(f"=== 金利スクレイピング開始: {len(banks)}行 ===")
    all_rates = []
    success = 0

    for bank in banks:
        print(f"\n--- {bank['name']} ({bank['id']}) ---")

        html = await fetch_page_html(bank["url"], bank["id"])
        if not html or len(html) < 500:
            continue

        markdown = html_to_markdown(html)
        print(f"  Markdown: {len(markdown)} chars")

        if len(markdown) < 30:
            print(f"  [SKIP] コンテンツ不足")
            continue

        prompt = build_llm_prompt(markdown, bank)
        result = await extract_with_llm(prompt)

        if result and "rates" in result and len(result["rates"]) > 0:
            all_rates.append(result)
            success += 1
            print(f"  [EXTRACTED] {len(result['rates'])} rate(s)")
        else:
            print(f"  [NO DATA]")

        time.sleep(interval)

    # 保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_rates, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完了: {success}/{len(banks)}行 ===")

    # Worker更新
    if all_rates:
        await update_worker(all_rates)


if __name__ == "__main__":
    asyncio.run(main())
