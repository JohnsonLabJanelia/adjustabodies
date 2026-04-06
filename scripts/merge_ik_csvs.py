#!/usr/bin/env python3
"""Merge per-animal IK CSVs into one file with animal column."""
import os, sys

input_dir = sys.argv[1]   # e.g. /path/to/qpos_v4/
output = sys.argv[2]      # e.g. /path/to/qpos_v4.csv

animals = ['captain', 'emilie', 'heisenberg', 'mario', 'remy']

with open(output, 'w') as out:
    out.write("# GREEN v4 per-animal IK export (merged)\n")
    header_written = False
    total = 0
    for animal in animals:
        path = os.path.join(input_dir, f'qpos_{animal}.csv')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            continue
        n = 0
        with open(path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if line.startswith('trial,'):
                    if not header_written:
                        out.write(line.replace('trial,frame,', 'trial,frame,animal,'))
                        header_written = True
                    continue
                # Insert animal after trial,frame
                comma1 = line.index(',')
                comma2 = line.index(',', comma1 + 1)
                out.write(line[:comma2] + ',' + animal + line[comma2:])
                n += 1
        total += n
        print(f"  {animal}: {n} frames")
    print(f"Total: {total} frames -> {output}")
