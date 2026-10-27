#!/usr/bin/env python3
"""Read-only Wolfram notebook inventory. Never evaluates notebook expressions.

Usage: python inspect_notebook.py 'new(1).nb' --out-dir notebook_audit
Produces index.csv and input_cells.json. Line numbers refer to the source file.
The original BoxData is preserved; no claim of a full Mathematica parser is made.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def closing_bracket(source, start):
    """Find balanced closing ] while respecting strings and nested (* comments *)."""
    depth, comment, quoted, escaped = 0, 0, False, False
    i = start
    while i < len(source):
        c = source[i]
        pair = source[i:i+2]
        if comment:
            if pair == '(*': comment += 1; i += 2; continue
            if pair == '*)': comment -= 1; i += 2; continue
        elif quoted:
            if escaped: escaped = False
            elif c == '\\': escaped = True
            elif c == '"': quoted = False
        elif pair == '(*': comment = 1; i += 2; continue
        elif c == '"': quoted = True
        elif c == '[': depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0: return i
        i += 1
    raise ValueError('Unterminated expression at character ' + str(start))


def inspect(path):
    data = path.read_bytes()
    source = data.decode('utf-8')
    # Exclude the duplicate, inert NotebookFileOutline cache at the end.
    source = source.split('(* Internal cache information *)')[0]
    result = []
    for match in re.finditer(r'^Cell\[', source, re.M):
        begin = match.start()
        end = closing_bracket(source, begin+4)
        cell = source[begin:end+1]
        if cell.startswith('Cell[CellGroupData['): continue
        if cell.startswith('Cell[BoxData['):
            box_end = closing_bracket(cell, len('Cell[BoxData'))
            style_match = re.match(r',\s*"([^"]+)"', cell[box_end+1:])
        else:
            style_match = re.match(r'Cell\["(?:\\.|[^"\\])*",\s*"([^"]+)"', cell, re.S)
        if not style_match or style_match[1] not in ('Input', 'Text', 'Section', 'Subsection', 'Subsubsection'): continue
        label = re.search(r'CellLabel->\s*"([^"]*)"', cell)
        uuid = re.search(r'ExpressionUUID->"([^"]*)"', cell)
        result.append(dict(index=len(result)+1, style=style_match[1], line=source.count('\n',0,begin)+1,
                           label=label[1] if label else '', uuid=uuid[1] if uuid else '', raw=cell))
    return result, hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('notebook_audit'))
    args = parser.parse_args()
    result, digest = inspect(args.notebook)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/'input_cells.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.out_dir/'index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index','style','line','label','uuid'])
        writer.writeheader()
        for row in result: writer.writerow({k:row[k] for k in writer.fieldnames})
    metadata = {'input':args.notebook.name,'sha256':digest,'retained_cells':len(result), 'execution':'none; static inspection only'}
    (args.out_dir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata))

if __name__ == '__main__': main()
