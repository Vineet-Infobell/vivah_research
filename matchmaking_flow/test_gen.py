import json

print("Loading approaches...")
approaches = []

# Load all 7 approaches
for i in [1, 2, 3, 4, 5, 6, 8]:
    try:
        with open(f"approach_{i}_results.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            approaches.append({"id": i, "data": data})
            print(f"✅ Loaded Approach {i}")
    except:
        print(f"⚠️ Approach {i} not found")

print(f"\nTotal approaches loaded: {len(approaches)}")

for app in approaches:
    print(f"\nApproach {app['id']}:")
    print(f"  NDCG: {app['data']['summary']['average_ndcg']:.3f}")
    print(f"  Latency: {app['data']['summary']['average_latency_ms']:.0f}ms")
