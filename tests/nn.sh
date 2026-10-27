#!/bin/sh
# compression + kernel cost per override set: nn histogram = nearest grain distance <0.5r,<1.2r,<1.8r,>=1.8r
scene=$1; secs=$2; shift 2
for set in "$@"; do
  timeout 150 node tests/probe.mjs "$scene" "set=$set" 64000 "$secs" 2>&1 | tail -1 | python -c "
import sys,json
line=sys.stdin.read(); t,_,js=line.partition(' ')
try:
  d=json.loads(js); print('%-30s'%'$set', t, 'nn',d['nn_lt05_lt12_lt18_ok'],'v50',d['v50'],'floor',d['onFloor'],'torn',d['torn'],'minYc',d['minYCloth'])
except Exception as e: print('$set', 'ERR', line[:200])
"
done
