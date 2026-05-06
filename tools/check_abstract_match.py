#!/usr/bin/env python3
"""Compare abstract in paper_c_v1.tex against locked abstract in
NEURIPS_SUBMISSIONS.md P2."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
paper = (ROOT / 'paper' / 'drafts' / 'paper_c_v1.tex').read_text(encoding='utf-8')
sub = (ROOT / 'paper' / 'drafts' / 'NEURIPS_SUBMISSIONS.md').read_text(encoding='utf-8')

m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', paper, re.DOTALL)
paper_abstract = m.group(1).strip()


def detex(s):
    s = re.sub(r'\$\\Mc\$', 'M_c', s)
    s = re.sub(r'\\Mc', 'M_c', s)
    s = re.sub(r'\$([^$]+)\$', r'\1', s)
    s = re.sub(r'\\emph\{([^}]+)\}', r'\1', s)
    s = s.replace('~', ' ').replace('---', '-').replace('--', '-')
    s = s.replace('M_{\\mathrm{c}}', 'M_c')
    s = re.sub(r'\\\\', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


paper_clean = detex(paper_abstract)
print(f'paper abstract: {len(paper_clean.split())} words, {len(paper_clean)} chars')

p2_start = sub.find('# P2')
p2_end = sub.find('# P3', p2_start)
p2_section = sub[p2_start:p2_end]
abs_header = p2_section.find('## Abstract*')
hunt = p2_section[abs_header:]
abs_match = re.search(r'```[a-zA-Z]*\n(.*?)\n```', hunt, re.DOTALL)
if not abs_match:
    print('Could not find locked abstract code-fence in P2 section')
    raise SystemExit(1)

locked = abs_match.group(1).strip()
locked_clean = re.sub(r'\s+', ' ', locked).strip()
print(f'locked abstract: {len(locked_clean.split())} words, {len(locked_clean)} chars')
print()

p_words = paper_clean.split()
l_words = locked_clean.split()

if locked_clean.lower() == paper_clean.lower():
    print('IDENTICAL after normalization')
else:
    diverged = False
    for i, (a, b) in enumerate(zip(p_words, l_words)):
        if a.lower().rstrip('.,;:') != b.lower().rstrip('.,;:'):
            print(f'first divergence at word {i}:')
            print(f'  paper:   {a!r}')
            print(f'  locked:  {b!r}')
            print(f'  paper context:   ...{" ".join(p_words[max(0,i-3):i+5])}...')
            print(f'  locked context:  ...{" ".join(l_words[max(0,i-3):i+5])}...')
            diverged = True
            break
    if not diverged:
        if len(p_words) != len(l_words):
            print(f'lengths differ: paper={len(p_words)} words, locked={len(l_words)} words')
            shorter = min(len(p_words), len(l_words))
            tail_p = p_words[shorter:]
            tail_l = l_words[shorter:]
            if tail_p:
                print(f'  paper tail:  {" ".join(tail_p[:10])}')
            if tail_l:
                print(f'  locked tail: {" ".join(tail_l[:10])}')