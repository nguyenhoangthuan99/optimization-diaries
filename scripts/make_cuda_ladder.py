#!/usr/bin/env python3
"""Render the recorded FP32 matmul ladder as separate, size-specific charts.

Source: posts/03-cuda-matmul-blackwell.md, 'The ladder, end to end'.
Run: python3 scripts/make_cuda_ladder.py (requires rsvg-convert).
"""
from pathlib import Path
from html import escape
import subprocess

OUT = Path(__file__).resolve().parents[1] / 'figures/03-cuda-matmul'
STAGES = [
    ('0 · Naive', 'Uncoalesced', '1 output/thread'),
    ('1 · Coalesced', 'Global accesses', '1 output/thread'),
    ('2 · Shared tile', '32×32 block tile', '1 output/thread'),
    ('3 · Register tile', '2×2 per thread', 'Shared-data reuse'),
    ('4 · Tuned', '64×128 block tile', '8×8/thread · BK=32'),
    ('5 · Async pipeline', 'cp.async · 2 buffers', '8×8/thread · BK=16'),
    ('6 · Retuned', 'cp.async · 2 buffers', '8×4/thread · BK=32'),
    ('cuBLAS', 'FP32 reference', 'Same matrix size'),
]
DATA = {2048: [0.83,6.15,8.53,24.85,39.55,50.05,54.3,63.78],
        4096: [0.83,6.59,8.55,25.68,50.72,63.40,65.0,77.96]}
W,H = 1440,760

def panel(size):
    vals = DATA[size]
    color = '#2563eb' if size == 2048 else '#d97706'
    bits = ['<rect width="1440" height="760" fill="white"/>']
    def text(x,y,s,fs=16,fill='#475569',weight=400,anchor='middle'):
        bits.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" fill="{fill}" font-weight="{weight}">{escape(s)}</text>')
    text(720,43,'CUDA FP32 Matrix Multiplication: Naive to cuBLAS on NVIDIA Blackwell',30,'#0f172a',700)
    text(720,75,f'RTX PRO 6000 · matmul memory-hierarchy optimization walk · M = N = K = {size}',18)
    text(720,105,f'Labels: TFLOP/s and % of this size’s cuBLAS baseline ({vals[-1]:.2f} TFLOP/s)',17)
    left,right,base,top = 90,1390,565,165
    scale=(base-top)/120
    for tick in range(0,121,20):
        y=base-tick*scale
        bits.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#e2e8f0"/>')
        text(left-14,y+6,str(tick),15,anchor='end')
    text(left,143,'TFLOP/s',16,anchor='start')
    peak=base-117*scale
    bits.append(f'<line x1="{left}" y1="{peak}" x2="{right}" y2="{peak}" stroke="#94a3b8" stroke-dasharray="7 5"/>')
    text(right,peak+22,'Theoretical FP32 peak: 117 TFLOP/s',15,anchor='end')
    for i,(v,labels) in enumerate(zip(vals,STAGES)):
        x=left+(i+.5)*(right-left)/8
        y=base-v*scale
        fill='#475569' if i==7 else color
        bits.append(f'<rect x="{x-49}" y="{y}" width="98" height="{v*scale}" rx="3" fill="{fill}"/>')
        text(x,y-34,f'{v:.2f}',21,'#0f172a',700)
        text(x,y-12,f'{100*v/vals[-1]:.1f}%',17,fill,600)
        for j,label in enumerate(labels):
            text(x,596+24*j,label,16,'#0f172a' if j==0 else '#475569',700 if j==0 else 400)
    text(90,701,'Block tile = output tile per block · per-thread tile = register accumulators · BK = reduction-tile depth',16,anchor='start')
    text(90,730,'Recorded timings from the blog’s ladder table; percentages recomputed separately for each matrix size.',15,anchor='start')
    return '\n'.join(bits)

def save(name,body,height):
    path=OUT/f'{name}.svg'
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" font-family="Helvetica, Arial, sans-serif">\n{body}\n</svg>\n')
    subprocess.run(['rsvg-convert','-o',str(path.with_suffix('.png')),str(path)],check=True)

if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    for size in DATA:
        save(f'fig_ladder_{size}',panel(size),H)
    # Preserve the original path for existing links, but with separated panels.
    save('fig_ladder',panel(2048)+f'<g transform="translate(0,{H})">'+panel(4096)+'</g>',H*2)
