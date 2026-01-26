import argparse
import datetime
import re
import os

def update_planning(sprint_num, sprint_name):
    filepath = 'docs/neural_graph_planning.md'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    with open(filepath, 'r') as f:
        content = f.read()

    # Update Date in Header
    today = datetime.date.today().isoformat()
    # Regex covers "Última Actualización: YYYY-MM-DD..."
    content = re.sub(r'Última Actualización: \d{4}-\d{2}-\d{2}.*', f'Última Actualización: {today} (Sprint {sprint_num} completado - {sprint_name})', content)

    # Mark Sprint as completed in the table
    # Looking for line: | **Sprint X** | ... | Status |
    lines = content.splitlines()
    new_lines = []
    found = False
    for line in lines:
        if f"**Sprint {sprint_num}**" in line:
            # Check if already completed
            if "✅" not in line:
                # Split by pipe to isolate the last column (Status)
                parts = line.split('|')
                # Expected format: empty | **Sprint** | Name | Desc | Status | empty
                if len(parts) >= 5:
                    # Replace the status column (second to last element usually, or index 4)
                    # We'll just replace the text in the last non-empty segment
                    parts[-2] = " ✅ "
                    line = "|".join(parts)
                    found = True
        new_lines.append(line)
    
    with open(filepath, 'w') as f:
        f.write("\n".join(new_lines) + "\n") # Ensure trailing newline
    
    if found:
        print(f"Updated {filepath}: Marked Sprint {sprint_num} as completed.")
    else:
        print(f"Updated {filepath}: Header date updated. Sprint {sprint_num} row not found or already completed.")

def update_functional(sprint_num):
    filepath = 'docs/neural_graph_funcional.md'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    with open(filepath, 'r') as f:
        content = f.read()

    # Update Date
    today = datetime.date.today().isoformat()
    content = re.sub(r'Fecha: \d{4}-\d{2}-\d{2}', f'Fecha: {today}', content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Updated {filepath}: Updated date.")

def append_report(sprint_num, sprint_name, description):
    filepath = 'docs/consolidated_sprint_reports.md'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    today = datetime.date.today().isoformat()
    
    report_template = f"""
================================================================================
# Report: sprint_{sprint_num}_report.md
================================================================================

# Sprint {sprint_num}: {sprint_name} - Final Report

**Fecha:** {today}
**Estado:** ✅ Completado

---

## Objetivo del Sprint

> {description}

---

## Entregables

*(Auto-generated placeholder - Please fill details)*

### 1. Feature Name
*   Description...

---

## Tests

| Test Case | Descripción |
|-----------|-------------|
| `test_placeholder` | ... |

---

## Código Destacado

```rust
// ...
```
"""
    # Ensure we append to a new line
    with open(filepath, 'r') as f:
        file_content = f.read()
    
    prefix = "\n" if not file_content.endswith("\n") else ""

    with open(filepath, 'a') as f:
        f.write(prefix + report_template)
    print(f"Appended report template to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Update docs after sprint completion.')
    parser.add_argument('sprint', type=int, help='Sprint number')
    parser.add_argument('name', type=str, help='Sprint name')
    parser.add_argument('description', type=str, help='Sprint objective description')
    
    args = parser.parse_args()
    
    print(f"--- Processing Sprint {args.sprint}: {args.name} ---")
    update_planning(args.sprint, args.name)
    update_functional(args.sprint)
    append_report(args.sprint, args.name, args.description)
    print("--- Done ---")

if __name__ == "__main__":
    main()
