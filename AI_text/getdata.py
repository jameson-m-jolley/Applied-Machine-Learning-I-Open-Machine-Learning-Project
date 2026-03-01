import os
import time
from openai import OpenAI
from google import genai

# --- API SETUP ---
# Get keys at: 
# Groq: console.groq.com | GitHub: github.com/marketplace/models | Gemini: aistudio.google.com
GROQ_KEY = os.environ.get("GROQ_API_KEY")
GITHUB_KEY = os.environ.get("GITHUB_TOKEN")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

# --- DATA SETTINGS ---
JUNK_PROMPTS = [
    "Write a 300-word sci-fi story about a spaceship in a gravity well.",
    "Explain how a steam engine works in a technical, dry tone.",
    "Describe a rainy city in a very flowery, poetic style with many adjectives.",
    "Write a debate transcript between two robots about the meaning of life."
]

def get_meta(prompt):
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

def get_gpt(prompt):
    client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=GITHUB_KEY)
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

def get_gemini(prompt):
    client = genai.Client(api_key=GEMINI_KEY)
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return res.text

def save_sample(folder, index, text):
    path = f"AI_text/{folder}"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/sample_{index}.txt", "w") as f:
        f.write(text)

if __name__ == "__main__":
    for i, prompt in enumerate(JUNK_PROMPTS):
        print(f"Collecting batch {i}...")
        time.sleep(12)
        try:
            save_sample("meta", i, get_meta(prompt))
            ##save_sample("gpt", i, get_gpt(prompt))
            save_sample("gemini", i, get_gemini(prompt))
            print(f"Batch {i} saved. Sleeping to respect free tier limits...")
            time.sleep(2) # Prevent rate limiting
        except Exception as e:
            print(f"Error on batch {i}: {e}")