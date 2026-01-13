import sys
import requests
from bs4 import BeautifulSoup


def scrape_to_txt(url, output_file):
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    paragraphs = soup.find_all("p")
    text = "\n".join(
        p.get_text(strip=True)
        for p in paragraphs
        if p.get_text(strip=True)
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("  python scrape.py <URL> <output.txt>")
        sys.exit(1)

    url = sys.argv[1]
    output_file = sys.argv[2]

    scrape_to_txt(url, output_file)