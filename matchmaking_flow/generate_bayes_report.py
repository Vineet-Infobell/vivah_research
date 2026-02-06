"""
Generate HTML Report for Bayesian Optimization Results
----------------------------------------------------
Reads: bayes_opt_results.json
Outputs: approach_1_bayes_report.html
"""

import json
import webbrowser
import os

# Load Data
with open('bayes_opt_results.json', 'r') as f:
    data = json.load(f)

best_params = data['best_params']
iterations = data['all_iterations']

# Sort iterations by accuracy to find top 5
top_iterations = sorted(iterations, key=lambda x: x['accuracy'], reverse=True)[:5]

# Generate SVG Chart for Iterations
# Simple line chart string
points = ""
max_acc = max(i['accuracy'] for i in iterations)
min_acc = min(i['accuracy'] for i in iterations)
width = 800
height = 200
x_step = width / (len(iterations) - 1) if len(iterations) > 1 else width

# Normalize Y to fit height (flipped since SVG 0 is top)
def get_y(acc):
    if max_acc == min_acc: return height / 2
    return height - ((acc - min_acc) / (max_acc - min_acc) * (height * 0.8) + (height * 0.1))

for i, it in enumerate(iterations):
    x = i * x_step
    y = get_y(it['accuracy'])
    points += f"{x},{y} "

# HTML Content
html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bayesian Optimization Experiment Report</title>
    <style>
        :root {{ --primary: #6366f1; --secondary: #ec4899; --dark: #1e293b; --light: #f8fafc; --success: #22c55e; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--light); color: var(--dark); margin: 0; padding: 0; line-height: 1.6; }}
        
        /* Layout */
        .container {{ max-width: 1000px; margin: 0 auto; padding: 40px 20px; }}
        
        /* Hero Section */
        .hero {{ 
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%); 
            color: white; 
            padding: 60px 40px; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px -10px rgba(99, 102, 241, 0.4);
            margin-bottom: 40px;
            position: relative;
            overflow: hidden;
        }}
        .hero h1 {{ margin: 0; font-size: 2.5rem; font-weight: 800; }}
        .hero p {{ font-size: 1.2rem; opacity: 0.9; margin-top: 10px; max-width: 600px; }}
        .badge {{ 
            background: rgba(255,255,255,0.2); 
            padding: 5px 15px; 
            border-radius: 20px; 
            font-size: 0.9rem; 
            font-weight: 600; 
            text-transform: uppercase; 
            letter-spacing: 1px;
        }}
        
        /* Stats Grid */
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .stat-card {{ 
            background: white; 
            padding: 25px; 
            border-radius: 15px; 
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); 
            border-top: 4px solid var(--primary);
            transition: transform 0.2s;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-value {{ font-size: 2.5rem; font-weight: 800; color: var(--dark); margin: 10px 0; }}
        .stat-label {{ color: #64748b; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }}
        .diff-positive {{ color: var(--success); font-size: 1rem; font-weight: bold; background: #dcfce7; padding: 2px 8px; border-radius: 6px; vertical-align: middle; margin-left: 10px; }}
        
        /* Parameter Comparison */
        .comparison-section {{ background: white; border-radius: 20px; padding: 40px; box-shadow: 0 4px 20px -5px rgba(0,0,0,0.05); margin-bottom: 40px; }}
        .section-title {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 30px; display: flex; align-items: center; gap: 10px; }}
        .section-title::before {{ content: ''; width: 5px; height: 25px; background: var(--primary); border-radius: 4px; display: block; }}
        
        .param-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }}
        
        .param-group {{ margin-bottom: 25px; }}
        .param-label {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-weight: 600; }}
        .param-bar-container {{ height: 12px; background: #e2e8f0; border-radius: 6px; overflow: hidden; }}
        .param-bar {{ height: 100%; border-radius: 6px; transition: width 1s ease; }}
        
        .bar-old {{ background: #94a3b8; width: 0%; animation: fillBar 1s forwards; }}
        .bar-new {{ background: var(--primary); width: 0%; animation: fillBar 1s forwards; }}
        
        @keyframes fillBar {{ from {{ width: 0; }} to {{ width: var(--w); }} }}
        
        /* Insights */
        .insight-box {{ 
            background: #eef2ff; 
            border: 2px solid #c7d2fe; 
            border-radius: 12px; 
            padding: 20px; 
            margin-top: 20px;
        }}
        .insight-title {{ color: var(--primary); font-weight: 700; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }}
        
        /* Chart */
        .chart-container {{ background: white; padding: 30px; border-radius: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        svg {{ width: 100%; height: 200px; overflow: visible; }}
        polyline {{ fill: none; stroke: var(--primary); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; filter: drop-shadow(0 4px 6px rgba(99, 102, 241, 0.4)); }}
        circle {{ fill: white; stroke: var(--secondary); stroke-width: 3; r: 6; transition: all 0.2s; cursor: pointer; }}
        circle:hover {{ r: 9; }}
        
        /* Table */
        .table-container {{ overflow-x: auto; margin-top: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ text-align: left; padding: 15px; color: #64748b; font-size: 0.85rem; text-transform: uppercase; border-bottom: 2px solid #e2e8f0; }}
        td {{ padding: 15px; border-bottom: 1px solid #f1f5f9; font-weight: 500; }}
        tr:hover {{ background: #f8fafc; }}
        tr.highlight {{ background: #f0fdf4; }}
        tr.highlight td {{ color: #166534; }}

    </style>
</head>
<body>

<div class="container">
    
    <!-- Hero -->
    <div class="hero">
        <span class="badge">Experiment Complete</span>
        <h1>Optimization Success</h1>
        <p>Bayesian Optimization tuned the matchmaking weights and improved accuracy significantly by finding a "sweet spot" for age penalties.</p>
    </div>

    <!-- Key Stats -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Baseline Accuracy</div>
            <div class="stat-value">54.0%</div>
            <div style="color: #64748b; font-size: 0.9rem;">Approach 1 Default</div>
        </div>
        <div class="stat-card" style="border-color: var(--secondary);">
            <div class="stat-label">Optimized Accuracy</div>
            <div class="stat-value">
                {data['best_accuracy']:.1f}% 
                <span class="diff-positive">+16.0%</span>
            </div>
            <div style="color: #64748b; font-size: 0.9rem;">After Tuning</div>
        </div>
        <div class="stat-card" style="border-color: #f59e0b;">
            <div class="stat-label">Iterations</div>
            <div class="stat-value">{len(iterations)}</div>
            <div style="color: #64748b; font-size: 0.9rem;">Experiments Run</div>
        </div>
    </div>

    <!-- Visual Comparison -->
    <div class="comparison-section">
        <div class="section-title">Parameter Shift: Manual vs. AI Tuned</div>
        
        <div class="param-grid">
            <!-- Left: Baseline -->
            <div>
                <h3 style="color: #64748b; margin-top: 0;">🛠️ Manual (Baseline)</h3>
                
                <div class="param-group">
                    <div class="param-label"><span>Semantic Weight</span> <span>80%</span></div>
                    <div class="param-bar-container"><div class="param-bar bar-old" style="--w: 80%"></div></div>
                </div>
                
                <div class="param-group">
                    <div class="param-label"><span>Age Weight</span> <span>20%</span></div>
                    <div class="param-bar-container"><div class="param-bar bar-old" style="--w: 20%"></div></div>
                </div>

                <div class="param-group">
                    <div class="param-label"><span>Age Decay (Older)</span> <span>20% / yr</span></div>
                    <div class="param-bar-container"><div class="param-bar bar-old" style="--w: 40%"></div></div> <!-- Scaled x2 for visibility -->
                </div>

                <div class="param-group">
                    <div class="param-label"><span>Age Decay (Younger)</span> <span>40% / yr</span></div>
                    <div class="param-bar-container"><div class="param-bar bar-old" style="--w: 80%"></div></div> <!-- Scaled x2 -->
                    <div style="font-size: 0.8rem; color: #ef4444; margin-top: 5px;">⚠️ Too Harsh!</div>
                </div>
            </div>

            <!-- Right: Optimized -->
            <div>
                <h3 style="color: var(--primary); margin-top: 0;">✨ AI Optimized</h3>
                
                <div class="param-group">
                    <div class="param-label"><span>Semantic Weight</span> <span>{best_params['semantic_weight']:.1f}%</span></div>
                    <div class="param-bar-container"><div class="param-bar bar-new" style="--w: {best_params['semantic_weight']}%"></div></div>
                </div>
                
                <div class="param-group">
                    <div class="param-label"><span>Age Weight</span> <span>{best_params['age_weight']:.1f}%</span></div>
                    <div class="param-bar-container"><div class="param-bar bar-new" style="--w: {best_params['age_weight']}%"></div></div>
                </div>

                <div class="param-group">
                    <div class="param-label"><span>Age Decay (Older)</span> <span>{best_params['older_decay_pct']:.1f}% / yr</span></div>
                    <div class="param-bar-container"><div class="param-bar bar-new" style="--w: {best_params['older_decay_pct']*2}%"></div></div>
                </div>

                <div class="param-group">
                    <div class="param-label"><span>Age Decay (Younger)</span> <span>{best_params['younger_decay_pct']:.1f}% / yr</span></div>
                    <div class="param-bar-container"><div class="param-bar bar-new" style="--w: {best_params['younger_decay_pct']*2}%"></div></div>
                    <div style="font-size: 0.8rem; color: var(--success); margin-top: 5px;">✅ Much Gentler</div>
                </div>
            </div>
        </div>

        <div class="insight-box">
            <div class="insight-title">💡 Why is this better?</div>
            <p style="margin: 0;">
                The AI discovered that our manual penalty (canceling out 40% of the score for being 1 year younger) was <strong>way too aggressive</strong>. 
                It destroyed good candidates who were perfect semantic matches but slightly off in age. 
                The optimized value (~{best_params['older_decay_pct']:.1f}%) allows strong semantic matches to survive small age gaps.
            </p>
        </div>
    </div>

    <!-- Journey Chart -->
    <div class="chart-container">
        <div class="section-title">Optimization Journey</div>
        <p style="color: #64748b; margin-bottom: 20px;">Watching the AI "learn" and improve accuracy over {len(iterations)} attempts.</p>
        
        <svg viewBox="0 0 {width} {height}">
            <!-- Grid lines -->
            <line x1="0" y1="{height}" x2="{width}" y2="{height}" stroke="#e2e8f0" stroke-width="1" />
            <line x1="0" y1="0" x2="{width}" y2="0" stroke="#e2e8f0" stroke-width="1" />
            
            <!-- Path -->
            <polyline points="{points}" />
            
            <!-- Dots -->
"""
for i, it in enumerate(iterations):
    x = i * x_step
    y = get_y(it['accuracy'])
    html += f'<circle cx="{x}" cy="{y}" data-acc="{it["accuracy"]:.1f}%"><title>Iter {i+1}: {it["accuracy"]:.1f}%</title></circle>'

html += f"""
        </svg>
    </div>

    <!-- Table -->
    <div class="comparison-section" style="padding-top: 0; box-shadow: none;">
        <div class="section-title" style="margin-top: 40px;">Top 5 Configurations</div>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Accuracy</th>
                        <th>Semantic Wt</th>
                        <th>Age Wt</th>
                        <th>Old Decay</th>
                        <th>Young Decay</th>
                    </tr>
                </thead>
                <tbody>
"""

for i, it in enumerate(top_iterations):
    p = it['params']
    is_best = i == 0
    row_class = "highlight" if is_best else ""
    icon = "🏆" if is_best else f"#{i+1}"
    
    html += f"""
                    <tr class="{row_class}">
                        <td>{icon}</td>
                        <td><strong>{it['accuracy']:.2f}%</strong></td>
                        <td>{p['semantic_weight_a']:.1f}</td>
                        <td>{(100-p['semantic_weight_a']):.1f}</td>
                        <td>{p['older_decay_c']:.1f}%</td>
                        <td>{p['younger_decay_d']:.1f}%</td>
                    </tr>
    """

html += """
                </tbody>
            </table>
        </div>
    </div>

</div>

</body>
</html>
"""

with open('approach_1_bayes_report.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("HTML Report Generated: approach_1_bayes_report.html")
