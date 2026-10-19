from pathlib import Path
import json

for p in Path('scratch/research_next').rglob('*.ipynb'):
    n = json.loads(p.read_text())
    out = []
    for cell in n['cells']:
        out.append(''.join(cell.get('source', [])))
        for output in cell.get('outputs', []):
            out.append(''.join(output.get('text', [])))
            out.append(''.join(output.get('data', {}).get('text/plain', [])))
    dest = p.with_suffix('.txt')
    dest.write_text('\n\n'.join(out))
    print(dest)
