#!/usr/bin/env python3
"""Sanity-check labels/refs/citations in paper_c_v1.tex."""
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parent.parent / 'paper' / 'drafts' / 'paper_c_v1.tex'
s = P.read_text(encoding='utf-8')

LABEL_RE = re.compile(r'\\label\{([^}]+)\}')
REF_RE = re.compile(r'\\(?:ref|autoref|eqref)\{([^}]+)\}')
CITE_RE = re.compile(r'\\cite[a-z]*\{([^}]+)\}')
BIB_RE = re.compile(r'\\bibitem\{([^}]+)\}')
SEC_RE = re.compile(r'\\section\*?\{([^}]+)\}')

labels = set(LABEL_RE.findall(s))
refs = set(REF_RE.findall(s))
cite_keys = set()
for c in CITE_RE.findall(s):
    for k in c.split(','):
        cite_keys.add(k.strip())
bib_keys = set(BIB_RE.findall(s))
sections = SEC_RE.findall(s)

print(f'labels:      {len(labels)}')
print(f'refs:        {len(refs)}')
print(f'cite-keys:   {len(cite_keys)}')
print(f'bib-keys:    {len(bib_keys)}')
print()

dangling = refs - labels
print(f'refs without label: {sorted(dangling) if dangling else "none"}')
unused_lab = labels - refs
print(f'labels never referenced (OK if section anchors): {sorted(unused_lab)}')
undef_cite = cite_keys - bib_keys
print(f'cite keys without bibitem: {sorted(undef_cite) if undef_cite else "none"}')
unused_bib = bib_keys - cite_keys
print(f'bibitems never cited: {sorted(unused_bib) if unused_bib else "none"}')

print()
print(f'sections: {len(sections)}')
for sec in sections:
    print(f'  - {sec}')

print()
for env in ('document', 'table', 'figure', 'enumerate', 'itemize',
            'thebibliography', 'abstract', 'tabular', 'equation*'):
    b = s.count(f'\\begin{{{env}}}')
    e = s.count(f'\\end{{{env}}}')
    flag = 'OK' if b == e else 'MISMATCH'
    print(f'  {env:20s} begin={b:3d} end={e:3d}  {flag}')

print()
print(f'lines: {len(s.splitlines())}')
print(f'chars: {len(s)}')