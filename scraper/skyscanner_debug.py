"""
Diagnostic Skyscanner - Phase 2
Lance le browser en mode VISIBLE pour voir ce qui bloque.
Exécute ce fichier EN PREMIER avant le scraper principal.
"""

import asyncio
import json
from playwright.async_api import async_playwright

ORIGIN      = "CMN"
DESTINATION = "MAD"
DATE        = "260320"   # une date dans ~7 jours

intercepted = []

async def handle_response(response):
    url = response.url
    # Logger TOUTES les requêtes réseau pour voir ce qui part
    if response.status in [200, 204]:
        ct = response.headers.get("content-type", "")
        if "json" in ct:
            try:
                data = await response.json()
                intercepted.append({"url": url, "data": data})
                print(f"  [JSON] {url[:100]}")
            except:
                pass
    elif response.status in [403, 429, 503]:
        print(f"  [BLOCKED {response.status}] {url[:100]}")

async def main():
    url = (
        f"https://www.skyscanner.fr/transport/vols/"
        f"{ORIGIN.lower()}/{DESTINATION.lower()}/{DATE}/"
        f"?adultsv2=1&cabinclass=economy&childrenv2=&ref=home&rtn=0"
        f"&outboundaltsenabled=false&inboundaltsenabled=false&preferdirects=false"
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,   # VISIBLE - tu vas voir le browser s'ouvrir
            slow_mo=500,
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="fr-FR",
            viewport={"width": 1280, "height": 800},
        )

        # Patch anti-détection
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
        """)

        page = await context.new_page()
        page.on("response", handle_response)

        print(f"[→] Ouverture de : {url}\n")
        print("    Regarde le browser :")
        print("    - Tu vois une page normale ? → OK, le problème est dans le timing")
        print("    - Tu vois un captcha / page d'erreur ? → Anti-bot actif")
        print("    - Page blanche ? → JS bloqué\n")

        await page.goto(url, timeout=60000, wait_until="networkidle")

        # Screenshot pour diagnostic
        await page.screenshot(path="debug_screenshot.png", full_page=True)
        print("\n[SAVED] debug_screenshot.png - regarde cette image")

        # Titre de la page
        title = await page.title()
        print(f"[PAGE TITLE] {title}")

        # Contenu HTML brut (premier 2000 chars)
        html = await page.content()
        print(f"\n[HTML DEBUT]\n{html[:2000]}\n[...]\n")

        # Attendre encore 10s pour voir si les données arrivent
        print("[WAIT] Attente 10s supplémentaires pour le chargement des vols...")
        await page.wait_for_timeout(10000)

        # Sauvegarder tout ce qui a été intercepté
        with open("debug_intercepted.json", "w", encoding="utf-8") as f:
            # Sauvegarder seulement les URLs pour éviter fichier trop gros
            json.dump(
                [{"url": x["url"], "keys": list(x["data"].keys()) if isinstance(x["data"], dict) else str(type(x["data"]))}
                 for x in intercepted],
                f, indent=2
            )

        print(f"\n[RÉSULTAT] {len(intercepted)} réponses JSON interceptées")
        if intercepted:
            print("URLs interceptées :")
            for x in intercepted:
                print(f"  → {x['url'][:100]}")
                print(f"     clés JSON : {list(x['data'].keys()) if isinstance(x['data'], dict) else '?'}")
        else:
            print("AUCUNE réponse JSON → Skyscanner bloque avant même d'envoyer les requêtes")
            print("Solution : voir commentaires dans le code ci-dessous")

        await browser.close()

    # ─── SOLUTIONS selon le diagnostic ───
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOLUTIONS selon ce que tu observes :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAS 1 - Captcha visible dans le browser :
  → Utilise playwright-stealth :
     pip install playwright-stealth
     from playwright_stealth import stealth_async
     await stealth_async(page)

CAS 2 - Page normale mais 0 JSON intercepté :
  → Skyscanner a changé ses endpoints
  → Relance les DevTools dans ton vrai browser et copie
     les nouveaux noms d'endpoints (Network → XHR)

CAS 3 - Erreur 403/429 dans les logs :
  → Rate limiting → ajouter plus de délai
  → Ou utiliser un vrai cookie de session (voir CAS 4)

CAS 4 - Tout bloqué, rien ne passe :
  → Solution de secours : copier les cookies depuis
     ton browser Chrome (F12 → Application → Cookies)
     et les injecter dans Playwright

CAS 5 - Données dans debug_intercepted.json mais pas parsées :
  → La structure JSON a changé, il faut adapter le parseur
    """)

if __name__ == "__main__":
    asyncio.run(main())