import requests
from bs4 import BeautifulSoup
import nltk
import spacy

# Download required resources (only first time)
nltk.download('punkt')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Read URLs
with open('urls.txt', 'r') as f:
    urls = f.read().splitlines()

def extract_headlines(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = []

        for tag in soup.find_all(['h1', 'h2']):
            text = tag.get_text(strip=True)
            if 5 < len(text) < 100:
                headlines.append(text)
        return headlines
    except Exception as e:
        print(f"Failed to scrape {url} due to {e}")
        return []

all_headlines = []
for url in urls:
    headlines = extract_headlines(url)
    all_headlines.extend(headlines)

print("\n📰 Headlines:")
for line in all_headlines:
    print("•", line)

print("\n🧠 NLP Analysis using nltk + spaCy:")
for headline in all_headlines[:5]:  # limit to 5 for demo
    print(f"\nHeadline: {headline}")

    # Tokenize using nltk
    tokens = nltk.word_tokenize(headline)
    print(f"Tokens: {tokens}")

    # Named Entity Recognition using spaCy
    doc = nlp(headline)
    print("Named Entities:")
    for ent in doc.ents:
        print(f"  - {ent.text} ({ent.label_})")
