"""
Comprehensive Comparison Report Generator
------------------------------------------
Generates detailed comparison report for all matchmaking approaches.
"""

import json
from datetime import datetime

# Load all approach results
approaches = []

approach_configs = [
    (1, "approach_1_results.json", "Semantic + Gaussian Age"),
    (2, "approach_2_results.json", "Hard Filters + Semantic Search"),
    (3, "approach_3_results.json", "Hard Filters + Cross-Encoder"),
    (4, "approach_4_results.json", "Two-Stage (Semantic → Cross-Encoder)"),
    (5, "approach_5_results.json", "Minimal Filters + Cross-Encoder"),
    (6, "approach_6_results.json", "Age Range + Semantic + Gaussian Age"),
    (8, "approach_8_results.json", "Multi-Vector with Separate Embeddings"),
]

for app_id, filename, name in approach_configs:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            approaches.append({
                "id": app_id,
                "name": name,
                "data": data
            })
            print(f"✅ Loaded Approach {app_id}")
    except FileNotFoundError:
        print(f"⚠️ Approach {app_id} not found")

if not approaches:
    print("❌ No approach results found!")
    exit(1)

print(f"\n✅ Total {len(approaches)} approaches loaded\n")

# Find best performers
best_approach = max(approaches, key=lambda x: x['data']['summary']['average_ndcg'])
fastest_approach = min(approaches, key=lambda x: x['data']['summary']['average_latency_ms'])
most_precise = max(approaches, key=lambda x: x['data']['summary']['average_precision'])

# Generate HTML
html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Matchmaking Approaches - Comprehensive Comparison</title>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        
        .header {{
            background: white;
            border-radius: 20px;
            padding: 60px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 40px;
        }}
        .header h1 {{
            font-size: 3.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 15px;
        }}
        .header .subtitle {{ font-size: 1.3em; color: #666; margin-bottom: 10px; }}
        .header .date {{ font-size: 1em; color: #999; }}
        
        .section {{
            background: white;
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .section h2 {{
            color: #667eea;
            font-size: 2.2em;
            margin-bottom: 25px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }}
        
        .executive-summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .executive-summary h2 {{ color: white; border: none; margin-bottom: 30px; }}
        .executive-summary .highlight {{
            background: rgba(255,255,255,0.2);
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid white;
        }}
        .executive-summary .highlight strong {{ font-size: 1.2em; }}
        
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
        }}
        .comparison-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        .comparison-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .comparison-table tr:hover {{ background: #f8f9fa; }}
        .comparison-table .best {{
            background: #d1fae5;
            font-weight: bold;
            color: #065f46;
        }}
        .comparison-table .worst {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
        }}
        
        .key-findings {{
            background: #fef3c7;
            border-left: 5px solid #f59e0b;
            padding: 25px;
            border-radius: 10px;
            margin: 25px 0;
        }}
        .key-findings h4 {{
            color: #b45309;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .key-findings ul {{
            list-style: none;
            padding: 0;
        }}
        .key-findings li {{
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }}
        .key-findings li:before {{
            content: "●";
            position: absolute;
            left: 0;
            color: #f59e0b;
            font-size: 1.2em;
        }}
        
        @media print {{
            body {{ background: white; padding: 0; }}
            .section {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 Matchmaking Approaches</h1>
        <h2 style="color: #667eea; margin: 20px 0;">Comprehensive Comparison Report</h2>
        <p class="subtitle">Performance Analysis & Recommendations</p>
        <p class="date">Generated: {datetime.now().strftime('%B %d, %Y')}</p>
    </div>

    <div class="executive-summary">
        <h2>📋 Executive Summary</h2>
        <div class="highlight">
            <strong>🏆 Best Overall Performance:</strong><br>
            Approach {best_approach['id']} - {best_approach['name']}<br>
            NDCG: {best_approach['data']['summary']['average_ndcg']:.3f} | 
            Precision: {best_approach['data']['summary']['average_precision']:.1f}% | 
            Latency: {best_approach['data']['summary']['average_latency_ms']:.0f}ms
        </div>
        <div class="highlight">
            <strong>⚡ Fastest Approach:</strong><br>
            Approach {fastest_approach['id']} - {fastest_approach['name']}<br>
            Latency: {fastest_approach['data']['summary']['average_latency_ms']:.0f}ms
        </div>
        <div class="highlight">
            <strong>🎯 Most Precise:</strong><br>
            Approach {most_precise['id']} - {most_precise['name']}<br>
            Precision: {most_precise['data']['summary']['average_precision']:.1f}%
        </div>
    </div>

    <div class="section">
        <h2>📊 Performance Comparison</h2>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Approach</th>
                    <th>Name</th>
                    <th>NDCG</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Latency (ms)</th>
                </tr>
            </thead>
            <tbody>
"""

# Sort approaches by NDCG
sorted_approaches = sorted(approaches, key=lambda x: x['data']['summary']['average_ndcg'], reverse=True)
best_ndcg = sorted_approaches[0]['data']['summary']['average_ndcg']
worst_ndcg = sorted_approaches[-1]['data']['summary']['average_ndcg']

for app in sorted_approaches:
    data = app['data']['summary']
    ndcg = data['average_ndcg']
    
    # Highlight best and worst
    ndcg_class = ''
    if ndcg == best_ndcg:
        ndcg_class = 'best'
    elif ndcg == worst_ndcg:
        ndcg_class = 'worst'
    
    html += f"""
                <tr>
                    <td><strong>Approach {app['id']}</strong></td>
                    <td>{app['name']}</td>
                    <td class="{ndcg_class}">{ndcg:.3f}</td>
                    <td>{data['average_precision']:.1f}%</td>
                    <td>{data['average_recall']:.1f}%</td>
                    <td>{data['average_latency_ms']:.0f}ms</td>
                </tr>
"""

html += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>📈 Visual Comparison</h2>
        <div class="chart-container">
            <canvas id="ndcgChart" width="400" height="200"></canvas>
        </div>
        <div class="chart-container">
            <canvas id="latencyChart" width="400" height="200"></canvas>
        </div>
    </div>

    <div class="section">
        <div class="key-findings">
            <h4>🔍 Key Findings</h4>
            <ul>
"""

# Generate key findings
if best_approach['id'] == 2:
    html += """
                <li><strong>Approach 2 (Hard Filters + Semantic)</strong> delivers the best overall performance with high NDCG and balanced speed</li>
"""
if fastest_approach['data']['summary']['average_latency_ms'] < 1300:
    html += f"""
                <li><strong>Approach {fastest_approach['id']}</strong> is the fastest at {fastest_approach['data']['summary']['average_latency_ms']:.0f}ms, ideal for real-time matching</li>
"""

# Find if multi-vector is loaded
multi_vector_app = next((a for a in approaches if a['id'] == 8), None)
if multi_vector_app:
    html += f"""
                <li><strong>Multi-Vector Approach (8)</strong> shows inconsistent performance (NDCG: {multi_vector_app['data']['summary']['average_ndcg']:.3f}) - works well for some professions but fails for others</li>
"""

html += """
                <li>Simple approaches consistently outperform complex multi-attribute decomposition</li>
                <li>Location as semantic matching (not hard filter) provides better flexibility</li>
            </ul>
        </div>
    </div>

    <div class="section">
        <h2>💡 Recommendations</h2>
        <div style="background: #eff6ff; border-left: 5px solid #3b82f6; padding: 25px; border-radius: 10px;">
            <h4 style="color: #1e40af; margin-bottom: 15px;">For Production Implementation:</h4>
            <ul style="line-height: 2;">
"""

html += f"""
                <li><strong>Primary Recommendation:</strong> Implement Approach {best_approach['id']} - {best_approach['name']}</li>
                <li><strong>Reasoning:</strong> Best NDCG ({best_approach['data']['summary']['average_ndcg']:.3f}) with reasonable latency ({best_approach['data']['summary']['average_latency_ms']:.0f}ms)</li>
"""

if best_approach['id'] == 2:
    html += """
                <li><strong>Configuration:</strong> Gender + Religion + Age Range filters, then pure semantic ranking</li>
                <li><strong>Advantages:</strong> Simple, fast, consistent across all scenarios</li>
"""

html += """
            </ul>
        </div>
    </div>
</div>

<script>
// NDCG Comparison Chart
const ndcgCtx = document.getElementById('ndcgChart').getContext('2d');
new Chart(ndcgCtx, {
    type: 'bar',
    data: {
        labels: [""" + ', '.join([f"'Approach {a['id']}'" for a in sorted_approaches]) + """],
        datasets: [{
            label: 'NDCG Score',
            data: [""" + ', '.join([f"{a['data']['summary']['average_ndcg']:.3f}" for a in sorted_approaches]) + """],
            backgroundColor: 'rgba(102, 126, 234, 0.8)',
            borderColor: 'rgba(102, 126, 234, 1)',
            borderWidth: 2
        }]
    },
    options: {
        responsive: true,
        plugins: {
            title: {
                display: true,
                text: 'NDCG Score Comparison',
                font: { size: 18 }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1.0
            }
        }
    }
});

// Latency Comparison Chart
const latencyCtx = document.getElementById('latencyChart').getContext('2d');
new Chart(latencyCtx, {
    type: 'bar',
    data: {
        labels: [""" + ', '.join([f"'Approach {a['id']}'" for a in sorted_approaches]) + """],
        datasets: [{
            label: 'Latency (ms)',
            data: [""" + ', '.join([f"{a['data']['summary']['average_latency_ms']:.0f}" for a in sorted_approaches]) + """],
            backgroundColor: 'rgba(118, 75, 162, 0.8)',
            borderColor: 'rgba(118, 75, 162, 1)',
            borderWidth: 2
        }]
    },
    options: {
        responsive: true,
        plugins: {
            title: {
                display: true,
                text: 'Latency Comparison',
                font: { size: 18 }
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});
</script>
</body>
</html>
"""

# Save report
output_file = "COMPREHENSIVE_COMPARISON_REPORT.html"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✅ Report generated: {output_file}")
print(f"\n📊 Summary:")
print(f"   Best: Approach {best_approach['id']} (NDCG: {best_approach['data']['summary']['average_ndcg']:.3f})")
print(f"   Fastest: Approach {fastest_approach['id']} ({fastest_approach['data']['summary']['average_latency_ms']:.0f}ms)")
print(f"   Most Precise: Approach {most_precise['id']} ({most_precise['data']['summary']['average_precision']:.1f}%)")
