import json
import pandas as pd
import re

results = []

with open("shor_output_11.log", "r") as f:
    lines = f.read()

segments = lines.split("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

for segment in segments:
    segment = segment.strip()

    match = re.search(r"RUNNING SHOR: N=(\d+), a=(\d+), ver=(\d+)", segment)
    if match:
        current_N = int(match.group(1))
        current_a = int(match.group(2))
        current_version = int(match.group(3))
        continue
    if segment.startswith("{") and segment.endswith("}"):
        try:
            stats = json.loads(segment)
            results.append(
                {
                    "N": current_N,
                    "a": current_a,
                    "version": current_version,
                    "num_qubits": stats.get("num_qubits"),
                    "num_gates": stats.get("num_gates"),
                    "gate_types": stats.get("gate_types"),
                }
            )
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)

df = pd.DataFrame(results)
print(df)
df.to_csv("shor_stats_11.csv", index=False)
