import re

def extract_payload(text: str) -> str:
    """
    If `text` contains “The answer is” or “My question is” (any casing),
    returns whatever follows that phrase (stripped of leading/trailing whitespace).
    Otherwise returns `text` unchanged.
    """
    # Compile once for efficiency; the (?i) makes it case‑insensitive
    pattern = re.compile(
        r'(?i)\b(?:the answer is|my question is)\b.*',  # match marker + the rest of the line
        re.IGNORECASE | re.DOTALL                       # DOTALL if you want to span multiple lines
    )
    match = pattern.search(text)
    if match:
        # Group 1 is “anything after” our trigger phrase
        return match.group(0).strip()
    # Fallback
    return text

# —— Examples ——
if __name__ == "__main__":
    examples = [
        "My question is  What color is the ball?",
        "some intro… The ANSWER is   C",
        "No trigger here, just plain text."
    ]
    for ex in examples:
        print(f"Input: {ex!r}")
        print("Output:", extract_payload(ex))
        print("---")