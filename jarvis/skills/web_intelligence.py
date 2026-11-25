"""Web intelligence skills: Weather (Open-Meteo), Wikipedia, Live RSS News, Search & Reader."""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import urllib.parse
import webbrowser
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from bs4 import BeautifulSoup

from jarvis.core.safety import RiskLevel
from jarvis.skills.registry import SkillContext, SkillRegistry, SkillResult

try:
    import wikipedia  # type: ignore
except ImportError:
    wikipedia = None

_KNOWLEDGE_HEADERS = {
    "User-Agent": "JARVIS-AI-Voice-Assistant/2.0 (https://localhost; educational-assistant)"
}

_PREFIX_STRIP_RE = re.compile(
    r"^(?:can\s+you\s+|could\s+you\s+|please\s+|hey\s+|jarvis[\s,]+)?"
    r"(?:tell\s+(?:me\s+)?about|talk\s+about|give\s+me\s+(?:info|information|details|facts)\s+(?:about|on)|"
    r"info\s+(?:about|on)|information\s+(?:about|on)|details\s+(?:about|on)|what\s+do\s+you\s+know\s+about|"
    r"who\s+(?:is|was|are|were)|what\s+(?:is|was|are|were)(?:\s+the\s+meaning\s+of)?|"
    r"where\s+(?:is|was|are)|when\s+(?:was|did|is)|how\s+(?:does|do|did)|"
    r"explain(?:\s+to\s+me)?|describe|define|meaning\s+of|definition\s+of|about)\s+",
    re.IGNORECASE,
)


def _trim_sentences(text: str, max_sentences: int = 3, max_chars: int = 460) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    chosen: list[str] = []
    for part in parts[:max_sentences]:
        candidate = " ".join(chosen + [part])
        if chosen and len(candidate) > max_chars:
            break
        chosen.append(part)
    result = " ".join(chosen).strip()
    if len(result) > max_chars:
        result = result[: max_chars - 1].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return result


def clean_knowledge_topic(query: str) -> str:
    """Extract a clean search topic from a natural-language question."""
    q = (query or "").strip().strip("?.! ")
    # Apply prefix stripping up to twice (e.g. "please tell me about MS Dhoni")
    for _ in range(2):
        stripped = _PREFIX_STRIP_RE.sub("", q).strip()
        if stripped == q or not stripped:
            break
        q = stripped
    q = re.sub(r"\s+(?:work|works|mean|means)$", "", q, flags=re.IGNORECASE).strip()
    q = re.sub(r"^(?:a|an|the)\s+", "", q, flags=re.IGNORECASE).strip()
    return q or (query or "").strip().strip("?.! ")


async def lookup_knowledge(query: str, sentences: int = 3) -> dict[str, Any] | None:
    """Zero-API-key live knowledge lookup: Dictionary + Wikipedia REST + DuckDuckGo Instant Answer."""
    raw = (query or "").strip()
    if not raw:
        return None
    low = raw.lower()
    topic = clean_knowledge_topic(raw)
    if not topic:
        return None

    is_define = bool(re.match(r"^(?:define|meaning\s+of|definition\s+of|what(?:'s|\s+is)\s+the\s+meaning\s+of)\b", low))

    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, headers=_KNOWLEDGE_HEADERS) as client:
        # 1. Free Dictionary API for single-word definitions
        if is_define and len(topic.split()) == 1:
            try:
                dr = await client.get(
                    f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(topic.lower())}"
                )
                if dr.status_code == 200:
                    entries = dr.json()
                    if isinstance(entries, list) and entries:
                        meanings = entries[0].get("meanings") or []
                        if meanings:
                            pos = meanings[0].get("partOfSpeech", "")
                            defs = meanings[0].get("definitions") or []
                            if defs and defs[0].get("definition"):
                                definition = defs[0]["definition"].strip()
                                prefix = f"{topic.capitalize()} ({pos}): " if pos else f"{topic.capitalize()}: "
                                return {
                                    "title": topic.capitalize(),
                                    "summary": f"{prefix}{definition}",
                                    "source": "Dictionary",
                                    "url": f"https://en.wiktionary.org/wiki/{urllib.parse.quote(topic.lower())}",
                                }
            except Exception:
                pass

        # 2. Wikipedia OpenSearch + REST v1 Summary (handles "MS Dhoni", abbreviations, nicknames cleanly!)
        try:
            search_resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": topic,
                    "limit": 4,
                    "namespace": 0,
                    "format": "json",
                },
            )
            candidates: list[str] = []
            if search_resp.status_code == 200:
                payload = search_resp.json()
                if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], list):
                    candidates = [str(t) for t in payload[1] if t]

            # Fallback to full MediaWiki search API if OpenSearch returns nothing (e.g. full questions)
            if not candidates:
                sr = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": topic,
                        "srlimit": 3,
                        "format": "json",
                    },
                )
                if sr.status_code == 200:
                    hits = sr.json().get("query", {}).get("search") or []
                    candidates = [str(h["title"]) for h in hits if h.get("title")]

            for title in candidates:
                sum_resp = await client.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}"
                )
                if sum_resp.status_code != 200:
                    continue
                data = sum_resp.json()
                if data.get("type") == "disambiguation":
                    continue
                extract = (data.get("extract") or "").strip()
                if len(extract) >= 40:
                    return {
                        "title": str(data.get("title") or title),
                        "summary": _trim_sentences(extract, max_sentences=max(1, min(5, int(sentences)))),
                        "source": "Wikipedia",
                        "url": str(data.get("content_urls", {}).get("desktop", {}).get("page", "")),
                    }
        except Exception:
            pass

        # 3. DuckDuckGo Instant Answer API fallback
        try:
            ddg = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": topic, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            if ddg.status_code == 200:
                d = ddg.json()
                abstract = (d.get("AbstractText") or d.get("Answer") or d.get("Definition") or "").strip()
                if not abstract:
                    related = d.get("RelatedTopics") or []
                    for item in related:
                        if isinstance(item, dict) and item.get("Text"):
                            abstract = str(item["Text"]).strip()
                            break
                if len(abstract) >= 25:
                    return {
                        "title": str(d.get("Heading") or topic.title()),
                        "summary": _trim_sentences(abstract, max_sentences=max(1, min(5, int(sentences)))),
                        "source": str(d.get("AbstractSource") or "Instant Knowledge"),
                        "url": str(d.get("AbstractURL") or ""),
                    }
        except Exception:
            pass

    return None


WMO_WEATHER_CODES: dict[int, str] = {
    0: "clear skies",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "moderate rain showers",
    82: "heavy rain showers",
    95: "thunderstorms",
}

SITE_SHORTCUTS: dict[str, str] = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "mail": "https://mail.google.com",
    "github": "https://github.com",
    "reddit": "https://www.reddit.com",
    "wikipedia": "https://en.wikipedia.org",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "gemini": "https://gemini.google.com",
    "perplexity": "https://www.perplexity.ai",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "linkedin": "https://www.linkedin.com",
    "instagram": "https://www.instagram.com",
    "insta": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "whatsapp": "https://web.whatsapp.com",
    "whatsapp web": "https://web.whatsapp.com",
    "telegram": "https://web.telegram.org",
    "discord": "https://discord.com/app",
    "netflix": "https://www.netflix.com",
    "prime video": "https://www.primevideo.com",
    "amazon prime": "https://www.primevideo.com",
    "hotstar": "https://www.hotstar.com",
    "jiohotstar": "https://www.hotstar.com",
    "spotify": "https://open.spotify.com",
    "twitch": "https://www.twitch.tv",
    "pinterest": "https://www.pinterest.com",
    "amazon": "https://www.amazon.in",
    "flipkart": "https://www.flipkart.com",
    "myntra": "https://www.myntra.com",
    "zomato": "https://www.zomato.com",
    "swiggy": "https://www.swiggy.com",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "drive": "https://drive.google.com",
    "google drive": "https://drive.google.com",
    "docs": "https://docs.google.com",
    "google docs": "https://docs.google.com",
    "sheets": "https://sheets.google.com",
    "google sheets": "https://sheets.google.com",
    "calendar": "https://calendar.google.com",
    "google calendar": "https://calendar.google.com",
    "meet": "https://meet.google.com",
    "google meet": "https://meet.google.com",
    "photos": "https://photos.google.com",
    "google photos": "https://photos.google.com",
    "translate": "https://translate.google.com",
    "google translate": "https://translate.google.com",
    "canva": "https://www.canva.com",
    "figma": "https://www.figma.com",
    "notion": "https://www.notion.so",
    "vercel": "https://vercel.com",
    "netlify": "https://www.netlify.com",
    "leetcode": "https://leetcode.com",
    "kaggle": "https://www.kaggle.com",
    "coursera": "https://www.coursera.org",
    "udemy": "https://www.udemy.com",
}


def open_url_on_os(url: str) -> bool:
    """Open a URL visibly in the user's OS default browser (explicit SW_SHOWNORMAL on Windows)."""
    if os.name == "nt":
        try:
            # Explicit SW_SHOWNORMAL (1) ensures newly opened browser tabs never inherit SW_HIDE
            rc = ctypes.windll.shell32.ShellExecuteW(None, "open", url, None, None, 1)
            if int(rc) > 32:
                return True
        except Exception:
            pass
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 1
            subprocess.Popen(
                ["cmd", "/c", "start", "", url],
                startupinfo=si,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            pass
    try:
        return bool(webbrowser.open_new_tab(url))
    except Exception:
        return False


def register_web_skills(registry: SkillRegistry) -> None:
    @registry.register(
        name="get_weather",
        description="Get live current weather conditions and temperature for any city using Open-Meteo (no API key needed).",
        category="web",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g., 'Tokyo', 'New York', 'Mumbai'). Uses default city if omitted.",
                }
            },
        },
        examples=["what is the weather in Tokyo", "check the weather", "temperature in London"],
    )
    async def get_weather(ctx: SkillContext, city: str = "") -> SkillResult:
        target_city = (city or ctx.config.default_city or "London").strip()
        async with httpx.AsyncClient(timeout=10.0) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": target_city, "count": 1, "language": "en", "format": "json"},
            )
            geo.raise_for_status()
            geo_data = geo.json()
            results = geo_data.get("results") or []
            if not results:
                return SkillResult(ok=False, message=f"I couldn't find weather coordinates for '{target_city}'.")

            loc = results[0]
            lat, lon = loc["latitude"], loc["longitude"]
            resolved_name = f"{loc['name']}, {loc.get('country', '')}".strip(", ")

            wx = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                },
            )
            wx.raise_for_status()
            cur = wx.json().get("current", {})

        temp = cur.get("temperature_2m")
        feels = cur.get("apparent_temperature")
        humidity = cur.get("relative_humidity_2m")
        wind = cur.get("wind_speed_10m")
        condition = WMO_WEATHER_CODES.get(int(cur.get("weather_code", 0)), "variable conditions")

        msg = (
            f"In {resolved_name}, it is currently {temp}°C (feels like {feels}°C) "
            f"with {condition}, humidity at {humidity}%, and wind at {wind} km/h."
        )
        return SkillResult(
            ok=True,
            message=msg,
            data={
                "location": resolved_name,
                "temperature_c": temp,
                "feels_like_c": feels,
                "humidity": humidity,
                "wind_kmh": wind,
                "condition": condition,
            },
        )

    @registry.register(
        name="search_wikipedia",
        description=(
            "Look up a concise factual summary ONLY when the user explicitly asks for external facts, "
            "a person, a concept, or a word definition. Pass a clean, specific topic query — NEVER call "
            "this on casual conversation, feelings, banter, or random single words mentioned in chat."
        ),
        category="web",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Clean, focused factual topic or person explicitly requested by the user.",
                },
                "sentences": {"type": "integer", "description": "Number of summary sentences (1-3 for voice)."},
            },
            "required": ["topic"],
        },
        examples=["tell about MS Dhoni", "search wikipedia for quantum computing", "who is Nikola Tesla"],
    )
    async def search_wikipedia(ctx: SkillContext, topic: str, sentences: int = 3) -> SkillResult:
        n = max(1, min(5, int(sentences)))
        hit = await lookup_knowledge(topic, sentences=n)
        if hit:
            return SkillResult(
                ok=True,
                message=hit["summary"],
                data={
                    "topic": hit["title"],
                    "summary": hit["summary"],
                    "source": hit["source"],
                    "url": hit["url"],
                },
            )
        # Fallback to python wikipedia library if direct REST was unavailable
        if wikipedia is not None:
            try:
                summary = wikipedia.summary(topic, sentences=n, auto_suggest=True)
                return SkillResult(ok=True, message=summary, data={"topic": topic, "summary": summary, "source": "Wikipedia"})
            except wikipedia.DisambiguationError as exc:
                options = ", ".join(exc.options[:5])
                return SkillResult(
                    ok=False,
                    message=f"'{topic}' could refer to several topics: {options}. Which one would you like?",
                    data={"options": exc.options[:5]},
                )
            except Exception:
                pass
        return SkillResult(ok=False, message=f"I couldn't find a concise encyclopedia summary for '{topic}'.")

    @registry.register(
        name="web_search_and_open",
        description=(
            "Open a specific website or run a focused search on Google, YouTube, or GitHub ONLY when the "
            "user explicitly asks to open a site or search the web. Pass a clean, specific query — NEVER "
            "call on casual chat, banter, or vague single-word remarks."
        ),
        category="web",
        risk=RiskLevel.MODERATE,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Focused search query, site shortcut (e.g. 'youtube'), or full URL.",
                },
                "platform": {
                    "type": "string",
                    "enum": ["google", "youtube", "github", "url"],
                    "description": "Search engine or direct URL mode.",
                },
            },
            "required": ["query"],
        },
        examples=["search google for FastAPI websockets", "play lo-fi beats on youtube", "open github"],
    )
    def web_search_and_open(ctx: SkillContext, query: str, platform: str = "google") -> SkillResult:
        raw = query.strip()
        plat = platform.lower().strip()
        if plat == "url" or raw.startswith(("http://", "https://")) or raw.lower() in SITE_SHORTCUTS:
            url = SITE_SHORTCUTS.get(raw.lower(), raw if raw.startswith(("http://", "https://")) else f"https://{raw}")
            if not ctx.defer_browser_open:
                open_url_on_os(url)
            label = raw.title() if raw.lower() in SITE_SHORTCUTS else url
            return SkillResult(
                ok=True,
                message=f"Opening {label} now.",
                data={"url": url, "label": label, "open_in_browser": True},
            )

        encoded = urllib.parse.quote_plus(raw)
        if plat == "youtube":
            url = f"https://www.youtube.com/results?search_query={encoded}"
        elif plat == "github":
            url = f"https://github.com/search?q={encoded}"
        else:
            url = f"https://www.google.com/search?q={encoded}"
        if not ctx.defer_browser_open:
            open_url_on_os(url)
        return SkillResult(
            ok=True,
            message=f"Opening {plat.title()} search for '{raw}'.",
            data={"url": url, "platform": plat, "label": f"{plat.title()}: {raw}", "open_in_browser": True},
        )

    @registry.register(
        name="get_top_news",
        description="Fetch the latest top headlines from live RSS news feeds (general or tech).",
        category="web",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["world", "tech", "science"],
                    "description": "News topic category.",
                },
                "limit": {"type": "integer", "description": "Number of headlines (1-8)."},
            },
        },
        examples=["give me the latest tech news", "read top headlines"],
    )
    async def get_top_news(ctx: SkillContext, category: str = "tech", limit: int = 5) -> SkillResult:
        feeds = {
            "tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
            "world": "https://feeds.bbci.co.uk/news/world/rss.xml",
            "science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        }
        url = feeds.get(category.lower().strip(), feeds["tech"])
        count = max(1, min(8, int(limit)))
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        root = ET.fromstring(resp.text)
        items: list[dict[str, str]] = []
        for item in root.findall(".//item")[:count]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            if title:
                items.append({"title": title, "link": link})

        if not items:
            return SkillResult(ok=False, message="No headlines were returned by the news feed.")

        spoken = " ".join(f"{idx + 1}. {it['title']}." for idx, it in enumerate(items))
        return SkillResult(
            ok=True,
            message=f"Here are the top {category} headlines: {spoken}",
            data={"category": category, "headlines": items},
        )

    @registry.register(
        name="fetch_webpage_summary",
        description="Fetch a web URL and extract its title and main readable text.",
        category="web",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP/HTTPS URL to fetch."}
            },
            "required": ["url"],
        },
        examples=["summarize https://example.com"],
    )
    async def fetch_webpage_summary(ctx: SkillContext, url: str) -> SkillResult:
        target = url.strip()
        if not target.startswith(("http://", "https://")):
            target = f"https://{target}"
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            resp = await client.get(target, headers={"User-Agent": "JARVIS-Voice-Assistant/2.0"})
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer"]):
            tag.decompose()
        title = (soup.title.string or target).strip() if soup.title else target
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
        excerpt = " ".join(paragraphs[:4])[:900]
        if not excerpt:
            excerpt = soup.get_text(" ", strip=True)[:600]
        return SkillResult(
            ok=True,
            message=f"{title}: {excerpt}",
            data={"url": target, "title": title, "excerpt": excerpt},
        )
