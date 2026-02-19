import wikipedia
import requests
import xml.etree.ElementTree as ET
def fetch_wikipedia(topic):
    wikipedia.set_lang("en")
    try:
        return wikipedia.summary(topic, sentences=5)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Topic ambiguous. Suggestions: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found."
    except Exception as e:
        return f"Error fetching Wikipedia data: {str(e)}"

# ArXiv
def fetch_arxiv(query):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results=3"
    response = requests.get(url)

    root = ET.fromstring(response.content)

    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []

    for entry in root.findall("atom:entry", namespace):
        title = entry.find("atom:title", namespace).text
        summary = entry.find("atom:summary", namespace).text

        papers.append({
            "title": title.strip(),
            "summary": summary.strip()
        })

    return papers

# News
def fetch_news(query):
    api_key = "YOUR_NEWS_API_KEY"   # replace with real key
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    return requests.get(url).json()

# Unified Loader
def load_datasets(sources, topic):
    data = {}

    if "wikipedia" in sources:
        data["wikipedia"] = fetch_wikipedia(topic)

    if "arxiv" in sources:
        data["arxiv"] = fetch_arxiv(topic)

    if "news" in sources:
        data["news"] = fetch_news(topic)

    return data
