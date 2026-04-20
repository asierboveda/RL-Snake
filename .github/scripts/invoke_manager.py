import os
import sys
import json
import subprocess
import re

def main():
    issue_body = os.environ.get('ISSUE_BODY', '')
    
    try:
        with open('.github/agents/manager.md', 'r', encoding='utf-8') as f:
            manager_prompt = f.read()
    except FileNotFoundError:
        print("Error: .github/agents/manager.md not found", file=sys.stderr)
        sys.exit(1)

    full_prompt = f"{manager_prompt}\n\nREQUERIMIENTO DEL ISSUE:\n{issue_body}"
    
    try:
        # Llamada headless a GitHub Copilot CLI
        result = subprocess.run(
            ['gh', 'copilot', 'explain', full_prompt],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing gh copilot: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    # Extracción estricta del JSON usando expresiones regulares
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', output, re.DOTALL | re.IGNORECASE)
    json_str = match.group(1).strip() if match else output.strip()

    try:
        parsed_json = json.loads(json_str)
        
        # Validación de contrato de datos
        if 'tasks' not in parsed_json:
            raise ValueError("Missing 'tasks' array in JSON object")
        for task in parsed_json['tasks']:
            if not all(k in task for k in ('worker_type', 'branch_name', 'instructions')):
                raise ValueError(f"Task missing required keys: {task}")
                
        # Salida limpia para GitHub Actions (stdout)
        print(json.dumps(parsed_json))
        
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON returned by Copilot. Raw output:\n{json_str}", file=sys.stderr)
        sys.exit(1)
    except ValueError as ve:
        print(f"Error in JSON structure validation: {ve}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()