from dotenv import load_dotenv
import os
import httpx

load_dotenv()

key = os.getenv("GROQ_API_KEY")

print("GROQ_API_KEY loaded:", bool(key))

if not key:
    print("ERROR: GROQ_API_KEY was not found in .env")
    exit()

response = httpx.get(
    "https://api.groq.com/openai/v1/models",
    headers={
        "Authorization": f"Bearer {key}"
    },
)

print("Status code:", response.status_code)
print(response.text[:5000])