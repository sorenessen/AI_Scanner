# pd_fingerprint.py — build n-gram fingerprints for CopyCat
# Usage:
#   python3 pd_fingerprint.py --src ./texts --out ./pd_fingerprints --name "gutenberg_mix" --n 5
import argparse, json, os, re, glob, collections
def toks(t): return re.findall(r"[a-zA-Z']+", t.lower())
def grams(ws, n): 
    return [" ".join(ws[i:i+n]) for i in range(len(ws)-n+1)] if len(ws) >= n else []
p = argparse.ArgumentParser()
p.add_argument("--src", required=True, help="dir of .txt files")
p.add_argument("--out", required=True, help="output dir for .json")
p.add_argument("--name", required=True, help='fingerprint name')
p.add_argument("--n", type=int, default=5, help="n-gram size (4–6 ok)")
args = p.parse_args()
counts = collections.Counter(); total = 0
for path in glob.glob(os.path.join(args.src, "*.txt")):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        w = toks(f.read()); g = grams(w, args.n); counts.update(g); total += len(g)
os.makedirs(args.out, exist_ok=True)
obj = {"name": args.name, "N": total, "ngrams": dict(counts)}
outp = os.path.join(args.out, f"{args.name}.json")
with open(outp, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False)
print(f"[ok] wrote {outp} (N={total:,}, unique={len(counts):,}, n={args.n})")
