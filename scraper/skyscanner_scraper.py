import asyncio
import json
import re
import pandas as pd
from datetime import datetime, timedelta
from playwright.async_api import async_playwright

try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

# config de base
ORIGIN      = "CMN"
DESTINATION = "MAD"
CABIN_CLASS = "economy"
ADULTS      = 1
CURRENCY    = "EUR"
OUTPUT_CSV  = "flights_raw.csv"
DAYS_AHEAD  = 30

# colle tes cookies Chrome ici (F12 → Application → Cookies → skyscanner.fr)
# sans ça skyscanner va bloquer les requêtes API
REAL_COOKIES = [
    # {"name": "traveller_context", "value": "...", "domain": ".skyscanner.fr"},
    # {"name": "__Secure-anon_token", "value": "...", "domain": ".skyscanner.fr"},
]

intercepted_data = []


async def handle_response(response):
    # on écoute uniquement les endpoints qui contiennent les vrais prix
    keywords = ["pricecalendar", "web-unified-search", "search-intent",
                "itineraries", "flights", "context"]

    if not any(k in response.url for k in keywords):
        return

    if response.status != 200:
        return

    try:
        ct = response.headers.get("content-type", "")
        if "json" not in ct:
            return

        data = await response.json()
        endpoint = next((k for k in keywords if k in response.url), "other")
        intercepted_data.append({
            "endpoint":   endpoint,
            "url":        response.url,
            "data":       data,
            "scraped_at": datetime.now().isoformat()
        })
        print(f"  [OK] {endpoint} → {response.url[:80]}")

    except Exception:
        pass


async def scrape_date(playwright, date_str: str) -> dict:
    url = (
        f"https://www.skyscanner.fr/transport/vols/"
        f"{ORIGIN.lower()}/{DESTINATION.lower()}/{date_str}/"
        f"?adultsv2={ADULTS}&cabinclass={CABIN_CLASS}"
        f"&childrenv2=&ref=home&rtn=0"
        f"&outboundaltsenabled=false&inboundaltsenabled=false&preferdirects=false"
    )

    browser = await playwright.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ]
    )

    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="fr-FR",
        timezone_id="Africa/Casablanca",
        viewport={"width": 1280, "height": 800},
        extra_http_headers={
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
        }
    )

    if REAL_COOKIES:
        await context.add_cookies(REAL_COOKIES)

    # skyscanner détecte les propriétés navigator typiques de Playwright
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver',  {get: () => undefined});
        Object.defineProperty(navigator, 'plugins',    {get: () => [1, 2, 3]});
        Object.defineProperty(navigator, 'languages',  {get: () => ['fr-FR', 'fr']});
        window.chrome = { runtime: {}, loadTimes: function(){} };
    """)

    page = await context.new_page()

    if STEALTH_AVAILABLE:
        await stealth_async(page)

    intercepted_data.clear()
    page.on("response", handle_response)

    print(f"\n[→] Date : {date_str}")

    result = {"date": date_str, "api_data": [], "dom_prices": [], "error": None}

    try:
        await page.goto(url, timeout=60000, wait_until="domcontentloaded")

        # skyscanner charge les vols en plusieurs requêtes successives
        # on attend jusqu'à 12s pour les avoir toutes
        for i in range(4):
            await page.wait_for_timeout(3000)
            if intercepted_data:
                print(f"  [DATA] {len(intercepted_data)} réponses après {(i+1)*3}s")
                break
            print(f"  [WAIT] {i+1}/4 — rien encore")

        result["api_data"] = intercepted_data.copy()

        # si l'API est bloquée on essaie de lire directement le HTML
        if not intercepted_data:
            print("  [FALLBACK] lecture DOM...")
            result["dom_prices"] = await extract_from_dom(page, date_str)

    except Exception as e:
        result["error"] = str(e)
        print(f"  [ERROR] {e}")

    finally:
        await browser.close()

    return result


async def extract_from_dom(page, date_str: str) -> list:
    # sélecteurs testés sur skyscanner.fr mars 2026
    # à adapter si le site change sa structure HTML
    selectors = [
        "[data-testid='price']",
        ".BpkText_bpk-text--heading-3__MTk4M",
        "[class*='Price_mainPriceContainer']",
        "span[aria-label*='€']",
    ]

    prices = []

    for selector in selectors:
        elements = await page.query_selector_all(selector)
        if not elements:
            continue

        for el in elements[:20]:
            text = await el.inner_text()
            match = re.search(r"[\d\s]+[,.]?\d*", text.replace('\xa0', ' '))
            if not match:
                continue
            try:
                price = float(match.group().replace(' ', '').replace(',', '.'))
                prices.append({
                    "date":        date_str,
                    "price":       price,
                    "currency":    CURRENCY,
                    "origin":      ORIGIN,
                    "destination": DESTINATION,
                    "scraped_at":  datetime.now().isoformat(),
                    "source":      "dom"
                })
            except ValueError:
                pass

        if prices:
            break

    return prices


def parse_all_data(results: list) -> pd.DataFrame:
    rows = []

    for r in results:
        date_str = r["date"]

        for item in r.get("api_data", []):
            data     = item["data"]
            endpoint = item["endpoint"]

            if "pricecalendar" in endpoint or "search-intent" in endpoint:
                for date_key, info in data.get("flights", {}).items():
                    if isinstance(info, dict) and "price" in info:
                        rows.append({
                            "date":         date_key or date_str,
                            "price":        info.get("price"),
                            "currency":     CURRENCY,
                            "airline":      "",
                            "stops":        None,
                            "duration_min": None,
                            "origin":       ORIGIN,
                            "destination":  DESTINATION,
                            "scraped_at":   item["scraped_at"],
                            "source":       "pricecalendar"
                        })

            elif "unified-search" in endpoint or "itineraries" in endpoint:
                itins = data.get("itineraries", {})
                for flight in (itins.get("results", []) if isinstance(itins, dict) else []):
                    try:
                        price_obj = flight.get("price", {})
                        leg       = (flight.get("legs") or [{}])[0]
                        carrier   = (leg.get("carriers", {}).get("marketing") or [{}])[0]
                        rows.append({
                            "date":         date_str,
                            "price":        price_obj.get("raw"),
                            "currency":     CURRENCY,
                            "airline":      carrier.get("name", ""),
                            "stops":        leg.get("stopCount"),
                            "duration_min": leg.get("durationInMinutes"),
                            "departure":    leg.get("departure", ""),
                            "arrival":      leg.get("arrival", ""),
                            "origin":       ORIGIN,
                            "destination":  DESTINATION,
                            "scraped_at":   item["scraped_at"],
                            "source":       "unified-search"
                        })
                    except Exception:
                        pass

        for row in r.get("dom_prices", []):
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


async def main():
    all_results = []

    async with async_playwright() as playwright:
        for i in range(DAYS_AHEAD):
            date_str = (datetime.today() + timedelta(days=i)).strftime("%y%m%d")
            result   = await scrape_date(playwright, date_str)
            all_results.append(result)

            print(f"  → API: {len(result['api_data'])} | DOM: {len(result['dom_prices'])} | err: {result['error']}")

            import random
            await asyncio.sleep(random.uniform(4, 8))

    df = parse_all_data(all_results)

    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        df.to_parquet(OUTPUT_CSV.replace(".csv", ".parquet"), index=False)
        print(f"\n[SAVED] {len(df)} lignes → {OUTPUT_CSV}")
        print(f"prix min: {df['price'].min()} | max: {df['price'].max()}")
    else:
        print("\n[VIDE] aucune donnée — lance skyscanner_debug.py pour voir pourquoi")

    with open("scraping_summary.json", "w") as f:
        json.dump([{
            "date":          r["date"],
            "api_responses": len(r["api_data"]),
            "dom_prices":    len(r["dom_prices"]),
            "error":         r["error"]
        } for r in all_results], f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())