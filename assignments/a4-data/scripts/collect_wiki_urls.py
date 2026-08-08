import json
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://en.wikipedia.org/w/api.php"
TARGET_NUM_URLS = 300

output_path = Path("data/quality/wiki_urls.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)

urls: set[str] = set()

while len(urls) < TARGET_NUM_URLS:
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": 10,
        "prop": "extlinks",
        "ellimit": "max",
        "format": "json",
        "formatversion": 2,
    }

    url = API_URL + "?" + urllib.parse.urlencode(params)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "cs336-a4-quality-collector/0.1"
        },
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        data = json.load(response)

    for page in data.get("query", {}).get("pages", []):
        for link in page.get("extlinks", []):
            external_url = link["url"]

            if external_url.startswith(("http://", "https://")):
                urls.add(external_url)

    print(f"Collected {len(urls)} URLs")

with open(output_path, "w") as f:
    for url in list(urls)[:TARGET_NUM_URLS]:
        f.write(url + "\n")

print(f"Wrote {TARGET_NUM_URLS} URLs to {output_path}")