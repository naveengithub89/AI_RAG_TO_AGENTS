from pathlib import Path
import yaml, re
from minsearch import Index

# Extract the podcast zip file
wget https://github.com/DataTalksClub/datatalksclub.github.io/archive/refs/heads/main.zip -O site.zip
unzip site.zip
mv datatalksclub.github.io-main/_podcast ./_podcast
rm -rf datatalksclub.github.io-main site.zip

def parse_yaml_front_matter(md_text: str):
    """Extract the YAML front matter block (between --- markers) and return it as a dict."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", md_text, flags=re.DOTALL)
    if not match:
        return {}
    yaml_block = match.group(1)
    try:
        return yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        return {}
    
def extract_transcript_docs(md_text: str, episode_slug: str):
    """Return one doc per transcript line with text, metadata, and timestamp."""
    meta = parse_yaml_front_matter(md_text)
    transcript = meta.get("transcript", []) or []

    title = meta.get("title")
    season = meta.get("season")
    episode = meta.get("episode")
    links = meta.get("links")

    docs = []
    for item in transcript:
        if isinstance(item, dict) and "line" in item:
            text = str(item["line"]).strip()
            if not text:
                continue
            sec = item.get("sec")
            timestamp = int(sec) if isinstance(sec, (int, float)) else None
            docs.append({
                "text": text,
                "title": title,
                "season": season,
                "episode": episode,
                "episode_slug": episode_slug,
                "youtube_link": links.get("youtube"),
                "timestamp": timestamp,
            })
    return docs


def sliding_window(items, size=30, overlap=15):
    step = size - overlap
    return [items[i:i+size] for i in range(0, len(items), step)]


# Ignoring the files that start with "_"
folder_path = Path("_podcast")
files = [f for f in folder_path.glob("*.md") if not f.name.startswith("_")]

chunks = []
for f in files:
    md_text = f.read_text(encoding="utf-8", errors="ignore")
    docs = extract_transcript_docs(md_text, episode_slug=f.stem)

    # chunk using sliding window
    for window in sliding_window(docs, size=30, overlap=15):
        combined_text = " ".join(d["text"] for d in window)
        chunks.append({
            "text": combined_text,
            "title": docs[0]["title"] if docs else "",
            "season": docs[0]["season"] if docs else "",
            "episode": docs[0]["episode"] if docs else "",
            "episode_slug": docs[0]["episode_slug"] if docs else ""
        })

print(f"Created {len(chunks)} chunks from {len(files)} podcast files.")


index = Index(text_fields=["text"])
index.fit(chunks)

def search_podcasts(query, n=5):
    return index.search(query, num_results=n)

def first_episode_for_query(query):
    results = search_podcasts(query, n=5)
    if not results:
        return "No matches found."

    # sort by season + episode number
    def sort_key(r):
        try:
            return (int(r.get("season", 0)), int(r.get("episode", 0)))
        except:
            return (9999, 9999)
    sorted_results = sorted(results, key=sort_key)
    first = sorted_results[0]

    return f"""
      **First relevant episode**
      **Title:** {first['title']}
      **Season:** {first['season']}, **Episode:** {first['episode']}
      **Episode ID:** {first['episode_slug']}
    """

query = "how do I make money with AI?"
print(first_episode_for_query(query))
