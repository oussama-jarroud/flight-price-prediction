import asyncio
import json
from playwright.async_api import async_playwright

ORIGIN = "CMN"
DESTINATION = "MAD"
DATE = "260320"

intercepted = []


async def handle_response(response):
    if response.status in [403, 429, 503]:
        print(f"  [BLOCKED {response.status}] {response.url[:100]}")
        return

    if response.status not in [200, 204]:
        return

    ct = response.headers.get("content-type", "")
    if "json" not in ct:
        return

    try:
        data = await response.json()
        intercepted.append({"url": response.url, "data": data})
        print(f"  [JSON] {response.url[:100]}")
    except Exception:
        pass


async def main():
    url = (
        f"https://www.skyscanner.fr/transport/vols/"
        f"{ORIGIN.lower()}/{DESTINATION.lower()}/{DATE}/"
        f"?adultsv2=1&cabinclass=economy&childrenv2=&ref=home&rtn=0"
        f"&outboundaltsenabled=false&inboundaltsenabled=false&preferdirects=false"
    )

    async with async_playwright() as p:
        # headless=False pour voir ce qui se passe visuellement
        browser = await p.chromium.launch(headless=False, slow_mo=500)

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="fr-FR",
            viewport={"width": 1280, "height": 800},
        )

        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
        """)

        page = await context.new_page()
        page.on("response", handle_response)

        print(f"[→] {url}\n")
        print("    page normale     → problème de timing")
        print("    captcha          → anti-bot actif, besoin des cookies")
        print("    page blanche     → JS bloqué\n")

        await page.goto(url, timeout=60000, wait_until="networkidle")

        await page.screenshot(path="debug_screenshot.png", full_page=True)
        print(f"\n[TITLE] {await page.title()}")

        # 10s de plus au cas où les vols chargent en retard
        print("[WAIT] 10s...")
        await page.wait_for_timeout(10000)

        # on sauvegarde juste les URLs + clés, pas les données complètes
        with open("debug_intercepted.json", "w", encoding="utf-8") as f:
            json.dump([{
                "url":  x["url"],
                "keys": list(x["data"].keys()) if isinstance(x["data"], dict) else str(type(x["data"]))
            } for x in intercepted], f, indent=2)

        print(f"\n{len(intercepted)} réponses JSON interceptées")
        for x in intercepted:
            print(f"  {x['url'][:100]}")
            print(f"    clés : {list(x['data'].keys()) if isinstance(x['data'], dict) else '?'}")

        if not intercepted:
            print("\nrien intercepté — skyscanner bloque tout")
            print("→ copie _pxvid et _px3 depuis Chrome F12 → Application → Cookies")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())