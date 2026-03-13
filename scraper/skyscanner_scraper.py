"""
Projet 11 - Scraper Skyscanner v2
Corrections anti-détection :
  1. playwright-stealth pour masquer les signaux d'automatisation
  2. Injection de cookies réels depuis ton browser Chrome
  3. Interception réseau plus large
  4. Fallback : scraping DOM si API non interceptée
"""

import asyncio
import json
import re
import time
import pandas as pd
from datetime import datetime, timedelta
from playwright.async_api import async_playwright

# ─── pip install playwright-stealth ───
try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False
    print("[WARN] playwright-stealth non installé. Exécute : pip install playwright-stealth")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
ORIGIN      = "CMN"
DESTINATION = "MAD"
CABIN_CLASS = "economy"
ADULTS      = 1
CURRENCY    = "EUR"
OUTPUT_CSV  = "flights_raw.csv"
DAYS_AHEAD  = 30

# ─── COOKIES RÉELS (optionnel mais recommandé) ───
# Comment les obtenir :
#   1. Ouvre skyscanner.fr dans Chrome normalement
#   2. F12 → Application → Cookies → https://www.skyscanner.fr
#   3. Copie les valeurs de : traveller_context, __Secure-anon_token, culture
#   4. Colle-les ici :
REAL_COOKIES = [
    # Exemple - remplace par tes vraies valeurs
    # {"name": "traveller_context", "value": "a8a830f4-9f68...", "domain": ".skyscanner.fr"},
    # {"name": "__Secure-anon_token", "value": "eyJhbGci...", "domain": ".skyscanner.fr"},
    # {"name": "culture", "value": "fr-FR", "domain": ".skyscanner.fr"},
]

# ─────────────────────────────────────────────
intercepted_data = []

async def handle_response(response):
    url = response.url
    # Intercepter TOUS les endpoints pertinents
    keywords = ["pricecalendar", "web-unified-search", "search-intent",
                 "itineraries", "flights", "context"]
    if any(k in url for k in keywords):
        try:
            if response.status == 200:
                ct = response.headers.get("content-type", "")
                if "json" in ct:
                    data = await response.json()
                    endpoint = next((k for k in keywords if k in url), "other")
                    intercepted_data.append({
                        "endpoint": endpoint,
                        "url": url,
                        "data": data,
                        "scraped_at": datetime.now().isoformat()
                    })
                    print(f"  [OK] {endpoint} → {url[:80]}")
        except Exception as e:
            pass   # réponse non-JSON, ignorer


async def scrape_date(playwright, date_str: str) -> dict:
    """Scrape une date. Retourne dict avec données API et/ou DOM."""
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
            "--disable-infobars",
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
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    # Injecter les cookies réels si disponibles
    if REAL_COOKIES:
        await context.add_cookies(REAL_COOKIES)

    # Patch anti-détection
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
        Object.defineProperty(navigator, 'languages', {get: () => ['fr-FR', 'fr']});
        window.chrome = { runtime: {}, loadTimes: function(){}, csi: function(){} };
    """)

    page = await context.new_page()

    # Appliquer stealth si disponible
    if STEALTH_AVAILABLE:
        await stealth_async(page)

    intercepted_data.clear()
    page.on("response", handle_response)

    print(f"\n[→] Date : {date_str}")

    result = {"date": date_str, "api_data": [], "dom_prices": [], "error": None}

    try:
        await page.goto(url, timeout=60000, wait_until="domcontentloaded")

        # Attente progressive pour le polling Skyscanner
        for wait_round in range(4):
            await page.wait_for_timeout(3000)
            if intercepted_data:
                print(f"  [DATA] {len(intercepted_data)} réponses après {(wait_round+1)*3}s")
                break
            print(f"  [WAIT] Round {wait_round+1}/4, données interceptées : {len(intercepted_data)}")

        result["api_data"] = intercepted_data.copy()

        # ─── FALLBACK : scraping DOM si API non interceptée ───
        if not intercepted_data:
            print("  [FALLBACK] Tentative scraping DOM...")
            dom_prices = await extract_from_dom(page, date_str)
            result["dom_prices"] = dom_prices
            if dom_prices:
                print(f"  [DOM] {len(dom_prices)} prix extraits depuis le HTML")

    except Exception as e:
        result["error"] = str(e)
        print(f"  [ERROR] {e}")
    finally:
        await browser.close()

    return result


async def extract_from_dom(page, date_str: str) -> list:
    """
    Fallback : extrait les prix directement depuis le DOM rendu.
    Adapte les sélecteurs selon ce que tu vois dans DevTools → Elements.
    """
    prices = []
    try:
        # Sélecteurs multiples pour couvrir différentes versions de Skyscanner
        selectors = [
            "[data-testid='price']",
            ".BpkText_bpk-text--heading-3__MTk4M",
            "[class*='Price_mainPriceContainer']",
            "[class*='price']",
            "span[aria-label*='€']",
        ]

        for selector in selectors:
            elements = await page.query_selector_all(selector)
            if elements:
                for el in elements[:20]:   # max 20 éléments
                    text = await el.inner_text()
                    # Extraire le nombre
                    match = re.search(r"[\d\s]+[,.]?\d*", text.replace('\xa0', ' '))
                    if match:
                        price_str = match.group().replace(' ', '').replace(',', '.')
                        try:
                            prices.append({
                                "date":        date_str,
                                "price":       float(price_str),
                                "currency":    CURRENCY,
                                "origin":      ORIGIN,
                                "destination": DESTINATION,
                                "scraped_at":  datetime.now().isoformat(),
                                "source":      "dom"
                            })
                        except ValueError:
                            pass
                if prices:
                    break   # sélecteur trouvé, inutile de continuer

    except Exception as e:
        print(f"  [DOM ERROR] {e}")

    return prices


# ─────────────────────────────────────────────
# PARSEURS JSON
# ─────────────────────────────────────────────
def parse_all_data(results: list) -> pd.DataFrame:
    rows = []

    for r in results:
        date_str = r["date"]

        # ─── Données API ───
        for item in r.get("api_data", []):
            data = item["data"]
            endpoint = item["endpoint"]

            # pricecalendar
            if "pricecalendar" in endpoint or "search-intent" in endpoint:
                flights = data.get("flights", {})
                for date_key, info in flights.items():
                    if isinstance(info, dict) and "price" in info:
                        rows.append({
                            "date":        date_key or date_str,
                            "price":       info.get("price"),
                            "currency":    CURRENCY,
                            "airline":     "",
                            "stops":       None,
                            "duration_min": None,
                            "origin":      ORIGIN,
                            "destination": DESTINATION,
                            "scraped_at":  item["scraped_at"],
                            "source":      "pricecalendar"
                        })

            # web-unified-search / itineraries
            elif "unified-search" in endpoint or "itineraries" in endpoint:
                itins = data.get("itineraries", {})
                flight_list = itins.get("results", []) if isinstance(itins, dict) else []
                for flight in flight_list:
                    try:
                        price_obj = flight.get("price", {})
                        legs = flight.get("legs", [])
                        leg = legs[0] if legs else {}
                        carriers = leg.get("carriers", {}).get("marketing", [{}])
                        carrier = carriers[0] if carriers else {}
                        rows.append({
                            "date":        date_str,
                            "price":       price_obj.get("raw"),
                            "currency":    CURRENCY,
                            "airline":     carrier.get("name", ""),
                            "stops":       leg.get("stopCount"),
                            "duration_min": leg.get("durationInMinutes"),
                            "departure":   leg.get("departure", ""),
                            "arrival":     leg.get("arrival", ""),
                            "origin":      ORIGIN,
                            "destination": DESTINATION,
                            "scraped_at":  item["scraped_at"],
                            "source":      "unified-search"
                        })
                    except Exception:
                        pass

        # ─── Données DOM (fallback) ───
        for row in r.get("dom_prices", []):
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def main():
    all_results = []

    async with async_playwright() as playwright:
        for i in range(DAYS_AHEAD):
            target = datetime.today() + timedelta(days=i)
            date_str = target.strftime("%y%m%d")

            result = await scrape_date(playwright, date_str)
            all_results.append(result)

            api_count = len(result["api_data"])
            dom_count = len(result["dom_prices"])
            print(f"  → API: {api_count} réponses | DOM: {dom_count} prix | erreur: {result['error']}")

            # Pause anti-bot (aléatoire entre 4-8s)
            import random
            await asyncio.sleep(random.uniform(4, 8))

    # Construire DataFrame
    df = parse_all_data(all_results)

    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        df.to_parquet(OUTPUT_CSV.replace(".csv", ".parquet"), index=False)
        print(f"\n[SAVED] {len(df)} lignes → {OUTPUT_CSV}")
        print(df.head(10).to_string())
        print(f"\nColonnes : {list(df.columns)}")
        print(f"Prix min : {df['price'].min()} | max : {df['price'].max()}")
    else:
        print("\n[VIDE] Aucune donnée collectée.")
        print("→ Lance d'abord skyscanner_debug.py pour diagnostiquer le blocage.")

    # Sauvegarde résumé JSON
    summary = [
        {
            "date": r["date"],
            "api_responses": len(r["api_data"]),
            "dom_prices": len(r["dom_prices"]),
            "error": r["error"]
        }
        for r in all_results
    ]
    with open("scraping_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[SAVED] scraping_summary.json")


if __name__ == "__main__":
    asyncio.run(main())