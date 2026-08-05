#!/usr/bin/env python3
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1.json")
artifact = json.loads(path.read_text(encoding="utf-8"))
contexts = artifact["contexts"]
checks = {
    "four independent contexts": len(contexts) == 4,
    "real vocabulary available": artifact["capabilities"]["vocabulary"] > 0,
    "all restored tokens identical": all(row["token_equal"] for row in contexts),
    "all restored logits bitwise-identical": all(row["logits_equal"] for row in contexts),
    "device-host-device round trip": artifact["host_round_trip"] is True,
    "cancellation preserved binding": artifact["cancellation_preserved_binding"] is True,
    "failed restore preserved binding": artifact["failed_promotion_preserved_binding"] is True,
}
failed = [name for name, passed in checks.items() if not passed]
print("# Cusco real-model inference integration report")
print(f"\nModel: `{artifact['model']['identity']}`")
print(f"Digest: `{artifact['model']['sha256']}`")
print(f"Vocabulary: {artifact['capabilities']['vocabulary']} tokens")
print(f"Elapsed: {artifact['elapsed_ms']} ms\n")
print("| Context | Prompt | Input token | Inferred next token | Checkpoint bytes | Exact after restore |")
print("|---:|---|---:|---:|---:|:---:|")
for row in contexts:
    exact = "PASS" if row["token_equal"] and row["logits_equal"] else "FAIL"
    print(f"| {row['context_index']} | `{row['prompt']}` | {row['continuation_input_token']} | {row['next_token']} | {row['checkpoint_bytes']} | {exact} |")
print("\n## Assertions")
for name, passed in checks.items():
    print(f"- {'PASS' if passed else 'FAIL'}: {name}")
if failed:
    raise SystemExit("integration proof failed: " + ", ".join(failed))
print("\nResult: PASS — llama.cpp performed real model evaluation and restored continuation exactly.")
