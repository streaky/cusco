#!/usr/bin/env python3
import json, pathlib, sys
report=json.load(open(sys.argv[1],encoding="utf-8"));root=pathlib.Path.cwd().resolve();failures=[];seen=[]
for file in report["data"][0]["files"]:
 path=pathlib.Path(file["filename"]).resolve()
 try: rel=path.relative_to(root)
 except ValueError: continue
 if not (len(rel.parts)>=3 and rel.parts[0]=="crates" and "src" in rel.parts): continue
 percent=float(file["summary"]["lines"]["percent"]);seen.append((str(rel),percent))
 if percent<80.0:failures.append((str(rel),percent))
for path,percent in sorted(seen):print(f"{percent:6.2f}% {path}")
if not seen:raise SystemExit("coverage report contained no crate source files")
if failures:raise SystemExit("per-file line coverage below 80%: "+", ".join(f"{p} ({n:.2f}%)" for p,n in failures))
