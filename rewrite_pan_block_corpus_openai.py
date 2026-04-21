#!/usr/bin/env python3
import os, json, time, argparse
from pathlib import Path
from openai import OpenAI

DELIM_LINE = "$$$"

SYSTEM = (
    "You are a careful rewriting assistant. Rewrite text to improve clarity while preserving meaning. "
    "Do not add new facts. Keep all names and numbers exactly unchanged."
)

USER_TMPL = """Rewrite the text to improve clarity and readability while preserving meaning.

Rules:
- Do NOT add new facts, names, or numbers.
- Keep all named entities and all numbers exactly unchanged.
- Keep English.
- Keep roughly the same length (±10%).
- Output ONLY the rewritten text.

TEXT:
{chunk}
"""

def rewrite(client: OpenAI, model: str, chunk: str, temperature: float, max_output_tokens: int) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(chunk=chunk)},
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return resp.output_text.strip()

def parse_blocks(raw: str):
    # blocks separated by blank lines (two \n in the generator script)
    # keep it robust: split on "\n\n" but drop empties
    blocks = [b for b in raw.split("\n\n") if b.strip()]
    return blocks

def split_block(block: str):
    lines = block.splitlines()
    # find the $$$ line
    try:
        idx = lines.index(DELIM_LINE)
    except ValueError:
        raise ValueError(f"Block missing '$$$' delimiter. Block starts:\n{block[:200]}")
    left = "\n".join(lines[:idx]).strip("\n")
    right = "\n".join(lines[idx+1:]).strip("\n")
    return left, right

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--progress", default=None)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max_output_tokens", type=int, default=1200)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    prog_path = Path(args.progress) if args.progress else Path(str(outfile) + ".progress.json")

    client = OpenAI()  # reads OPENAI_API_KEY

    raw = infile.read_text(encoding="utf-8", errors="replace")
    blocks = parse_blocks(raw)

    prog = {"done_blocks": 0}
    if prog_path.exists():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
    start = int(prog.get("done_blocks", 0))

    out_blocks = []
    if outfile.exists():
        out_raw = outfile.read_text(encoding="utf-8", errors="replace")
        out_blocks = parse_blocks(out_raw)
        start = max(start, len(out_blocks))

    print(f"Total blocks: {len(blocks)}")
    print(f"Resuming at block: {start}")
    print(f"Writing to: {outfile}")

    for i in range(start, len(blocks)):
        left, right = split_block(blocks[i])

        # rewrite left
        left_new = rewrite(client, args.model, left, args.temperature, args.max_output_tokens)
        time.sleep(args.sleep)

        # rewrite right
        right_new = rewrite(client, args.model, right, args.temperature, args.max_output_tokens)
        time.sleep(args.sleep)

        new_block = f"{left_new}\n{DELIM_LINE}\n{right_new}"
        out_blocks.append(new_block)

        # write incrementally with blank line between blocks (same as generator)
        outfile.write_text("\n\n".join(out_blocks) + "\n\n", encoding="utf-8")
        prog["done_blocks"] = i + 1
        prog_path.write_text(json.dumps(prog, indent=2), encoding="utf-8")

        if (i + 1) % 20 == 0:
            print(f"[OK] {i+1}/{len(blocks)}")

    print("Done.")

if __name__ == "__main__":
    main()
