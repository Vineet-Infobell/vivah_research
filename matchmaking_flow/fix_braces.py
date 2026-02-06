import re

with open('generate_comparison_report.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the HTML content string
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'html_content = """' in line or 'html_content += """' in line or 'html_content += f"""' in line:
        # Check if this line has CSS/HTML
        j = i
        while j < len(lines) and '"""' not in lines[j][1:]:  # Skip the opening """
            # Fix single braces in CSS
            if '{' in lines[j] or '}' in lines[j]:
                # Replace single { with {{ and single } with }}
                lines[j] = lines[j].replace('{', '{{').replace('}', '}}')
                # Fix any that were already doubled
                lines[j] = lines[j].replace('{{{{', '{{').replace('}}}}', '}}')
            j += 1

with open('generate_comparison_report.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("✅ Fixed braces")
