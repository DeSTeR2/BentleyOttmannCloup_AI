import sys
import time

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

MODEL = "gemini-3.8-flash"
PROMPT = (
    "The image shows line segments drawn on a coordinate grid. "
    "Return every segment as the coordinates of its two endpoints, "
    "read from the axis labels of the grid."
)


class Segment(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


def intersect(a, b):
    # a.p + t * r = b.p + u * s, intersection exists if 0 <= t, u <= 1
    rx, ry = a.x2 - a.x1, a.y2 - a.y1
    sx, sy = b.x2 - b.x1, b.y2 - b.y1
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-9:  # parallel or collinear, skipped
        return None
    qx, qy = b.x1 - a.x1, b.y1 - a.y1
    t = (qx * sy - qy * sx) / denom
    u = (qx * ry - qy * rx) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return a.x1 + t * rx, a.y1 + t * ry
    return None


def ask_gemini(path):
    image = open(path, "rb").read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    client = genai.Client()  # API key is taken from GEMINI_API_KEY
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=image, mime_type=mime), PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[Segment],
                    temperature=0,
                ),
            )
        except errors.APIError as e:
            if e.code != 429 or attempt == 2:
                raise
            print(f"429 rate limit, retry in {10 * (attempt + 1)} s...")
            time.sleep(10 * (attempt + 1))


def main():
    if len(sys.argv) != 2:
        print("usage: python main.py <image>")
        sys.exit(1)

    start = time.time()
    response = ask_gemini(sys.argv[1])
    segments = response.parsed
    print(f"Gemini ({MODEL}) answered in {time.time() - start:.1f} s")

    print(f"\nSegments found: {len(segments)}")
    for i, s in enumerate(segments):
        print(f"  S{i}: ({s.x1:g}, {s.y1:g}) - ({s.x2:g}, {s.y2:g})")

    print("\nIntersections:")
    count = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            p = intersect(segments[i], segments[j])
            if p:
                count += 1
                print(f"  S{i} x S{j}: ({p[0]:.2f}, {p[1]:.2f})")
    print(f"  total: {count}")

    usage = response.usage_metadata
    print(f"\nTokens: input {usage.prompt_token_count}, "
          f"output {usage.candidates_token_count}, total {usage.total_token_count}")


if __name__ == "__main__":
    main()
