"""
銀行金利スクレイピング + LLM抽出スクリプト
- 1st: Playwright (User-Agent/ヘッダー完全偽装)
- 2nd: httpx直接取得（軽量ページ用フォールバック）
- LLM: GitHub Models GPT-4o-mini
"""
import asyncio
import json
import os
import random
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

# リアルなChrome UA + ヘッダー
CHROME_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
HEADERS = {
    "User-Agent": CHROME_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


async def fetch_with_playwright(url: str, bank_id: str) -> str:
    """Playwright（完全なブラウザ偽装）"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        )
        context = await browser.new_context(
            user_agent=CHROME_UA,
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
            viewport={"width": 1920, "height": 1080},
            extra_http_headers={
                "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
                "Sec-CH-UA": '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="8"',
                "Sec-CH-UA-Mobile": "?0",
                "Sec-CH-UA-Platform": '"Windows"',
            }
        )
        # WebDriver検出を回避
        page = await context.new_page()
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'languages', { get: () => ['ja', 'en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        """)

        try:
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            if resp and resp.status >= 400:
                print(f"  [HTTP {resp.status}] {bank_id}")
                await browser.close()
                return ""
            # ページのJS実行を待つ
            await page.wait_for_timeout(random.randint(2000, 4000))
            html = await page.content()
            print(f"  [Playwright OK] {bank_id}: {len(html)} chars")
        except Exception as e:
            print(f"  [Playwright FAIL] {bank_id}: {str(e)[:80]}")
            html = ""
        finally:
            await browser.close()

    return html


async def fetch_with_httpx(url: str, bank_id: str) -> str:
    """httpx直接取得（フォールバック）"""
    async with httpx.AsyncClient(
        timeout=30,
        follow_redirects=True,
        headers=HEADERS,
        http2=True,
    ) as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                print(f"  [httpx OK] {bank_id}: {len(resp.text)} chars")
                return resp.text
            else:
                print(f"  [httpx {resp.status_code}] {bank_id}")
                return ""
        except Exception as e:
            print(f"  [httpx FAIL] {bank_id}: {str(e)[:80]}")
            return ""


async def fetch_page(url: str, bank_id: str) -> str:
    """Playwright → httpxの順で試行"""
    html = await fetch_with_playwright(url, bank_id)
    if not html or len(html) < 500:
        print(f"  → httpxフォールバック...")
        html = await fetch_with_httpx(url, bank_id)
    return html


def html_to_markdown(html: str, bank_name: str) -> str:
    """HTML -> Markdown。金利テーブルを優先抽出"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript", "iframe", "svg"]):
        tag.decompose()

    # テーブルを優先的に抽出（金利は大抵テーブルにある）
    tables = soup.find_all("table")
    rate_tables = []
    for table in tables:
        text = table.get_text()
        if any(kw in text for kw in ["金利", "利率", "%", "年率", "普通預金"]):
            rate_tables.append(str(table))

    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0

    # テーブルがあればテーブル優先
    if rate_tables:
        table_md = converter.handle("\n".join(rate_tables[:3]))
        if len(table_md) > 100:
            return table_md[:4000]

    # テーブルがなければ全体からキーワード周辺を抽出
    full_md = converter.handle(str(soup))
    if len(full_md) > 5000:
        lines = full_md.split("\n")
        relevant = []
        keywords = ["金利", "利率", "%", "年率", "普通預金", "優遇", "ステージ", "連携", "ハイブリッド", "マネーブリッジ", "コネクト"]
        for i, line in enumerate(lines):
            if any(kw in line for kw in keywords):
                start = max(0, i - 5)
                end = min(len(lines), i + 10)
                relevant.extend(lines[start:end])
        seen = set()
        deduped = []
        for line in relevant:
            if line not in seen:
                seen.add(line)
                deduped.append(line)
        full_md = "\n".join(deduped)

    return full_md[:4000]


def build_llm_prompt(markdown: str, bank_config: dict) -> str:
    terms_info = json.dumps(bank_config["terms"], ensure_ascii=False, indent=2)
    return f"""以下は「{bank_config['name']}」の金利ページのテキストです。

**普通預金金利**を全て抽出してください。

## 用語辞書:
{terms_info}

## ルール:
- 普通預金（相当商品含む）の金利のみ抽出
- 定期預金は除外
- 条件付きは条件を明記
- 年率(%)で統一
- データが見つからない場合は rates: [] で返す

## 出力（JSONのみ）:
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
      "condition": null or "条件",
      "condition_name": null or "名称",
      "limit_amount": null or "残高条件",
      "note": null
    }}
  ]
}}
```

## テキスト:
{markdown}"""


async def extract_with_llm(prompt: str, bank_id: str) -> dict:
    if not GITHUB_TOKEN:
        return {}

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GITHUB_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "銀行金利データの抽出。正確にJSON出力。データ不明なら rates:[] を返す。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
    }

    async with httpx.AsyncClient(timeout=90) as client:
        try:
            resp = await client.post(GITHUB_MODELS_URL, json=payload, headers=headers)
            if resp.status_code != 200:
                print(f"  [LLM {resp.status_code}] {bank_id}: {resp.text[:80]}")
                return {}

            content = resp.json()["choices"][0]["message"]["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content.strip())
            return result
        except Exception as e:
            print(f"  [LLM ERROR] {bank_id}: {e}")
            return {}


async def update_worker(all_rates: list):
    if not WORKER_URL or not CF_API_TOKEN:
        print("[SKIP] Worker update")
        return
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{WORKER_URL}/api/admin/update-rates",
                json={"rates": all_rates},
                headers={"Authorization": f"Bearer {CF_API_TOKEN}"}
            )
            print(f"[Worker] {resp.status_code}")
        except Exception as e:
            print(f"[Worker ERROR] {e}")


async def main():
    config = load_config()
    banks = config["banks"]

    print(f"=== 金利スクレイピング: {len(banks)}行 ===\n")
    all_rates = []
    success = 0
    failed_banks = []

    for bank in banks:
        print(f"--- {bank['name']} ---")

        html = await fetch_page(bank["url"], bank["id"])
        if not html or len(html) < 500:
            failed_banks.append(bank["name"])
            continue

        markdown = html_to_markdown(html, bank["name"])
        print(f"  MD: {len(markdown)} chars")

        if len(markdown) < 30:
            failed_banks.append(bank["name"])
            continue

        result = await extract_with_llm(
            build_llm_prompt(markdown, bank), bank["id"]
        )

        if result and "rates" in result and len(result["rates"]) > 0:
            all_rates.append(result)
            success += 1
            rates_info = ", ".join([f'{r["rate_percent"]}%' for r in result["rates"]])
            print(f"  ✓ {len(result['rates'])} rates: {rates_info}")
        else:
            failed_banks.append(bank["name"])
            print(f"  ✗ 抽出失敗")

        # ランダムな間隔（bot検出回避）
        await asyncio.sleep(random.uniform(3, 7))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_rates, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"成功: {success}/{len(banks)}行")
    if failed_banks:
        print(f"失敗: {', '.join(failed_banks)}")
    print(f"{'='*50}")

    if all_rates:
        await update_worker(all_rates)


if __name__ == "__main__":
    asyncio.run(main())
