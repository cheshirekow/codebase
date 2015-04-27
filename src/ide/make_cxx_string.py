#!/usr/bin/python
import sys

with open(sys.argv[1]) as f:
  for line in f.readlines():
    print r'"{}\n"'.format(line.rstrip().replace('"',r'\"'));
