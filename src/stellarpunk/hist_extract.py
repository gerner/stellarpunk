""" Tool to grab relevant entries from a debug history """

import sys
import argparse
import json
import gzip
import contextlib
import logging
from typing import Dict

import numpy as np

from stellarpunk import util

# This should be sized to select a single history entry given a timestamp, so
# it depends on the timestep used in the simulation.
TS_EPS = 1/30/2*10

def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logger = logging.getLogger(util.fullname(__name__))

    with contextlib.ExitStack() as context_stack:

        parser = argparse.ArgumentParser(description="")
        parser.add_argument("-i", "--input", nargs="?", type=str, default="-",
                help="file that has history in it, \"-\" for stdin. default \"-\"")
        parser.add_argument("-o", "--output", nargs="?", type=str, default="-",
                help="file to write output to, \"-\" for stdout. default \"-\"")
        parser.add_argument("eid", nargs="+", type=str,
                help="eids to extract")
        parser.add_argument("-t", "--timestamp", nargs="?", type=float, default=0.,
                help="the timestamp to extract")
        parser.add_argument("-l", "--location", nargs="?", type=str, default=None,
                help="a point around which to grab all entities at timestamp")
        parser.add_argument("--pdb", action="store_true")
        parser.add_argument("--tseps", type=float, default=TS_EPS,
                help="value smaller than any timestamp to select exactly one timestamp")

        args = parser.parse_args()

        if args.pdb:
            context_stack.enter_context(util.PDBManager())

        global_loc = None
        global_radius = 0
        if args.location:
            x,_,y = args.location.partition(",")
            x = float(x)
            y = float(y)
            global_loc = np.array((x,y))
            global_radius = 100000

        static_loc = []
        target_ts = {}
        for pattern in args.eid:
            if "," in pattern:
                _,_,l = pattern.partition(":")
                x,_,y = l.partition(",")
                x = float(x)
                y = float(y)
                static_loc.append((x,y))
            else:
                eid, _, ts = pattern.partition(":")
                if ts != "":
                    target_ts[eid] = float(ts)
                else:
                    target_ts[eid] = args.timestamp

        ts_eps = args.tseps

        if args.input == "-":
            fin = sys.stdin
        elif args.input.endswith(".gz"):
            fin = context_stack.enter_context(gzip.open(args.input, "rt"))
        else:
            fin = context_stack.enter_context(open(args.input, "rt"))

        if args.output == "-":
            fout = sys.stdout
        else:
            fout = context_stack.enter_context(open(args.output, "wt"))

        eid_matches:Dict[str, str] = {}
        eid_ts:Dict[str, float] = {}
        for line in fin:
            entry = json.loads(line)

            if entry["ts"] == 0:
                for loc in static_loc:
                    if np.linalg.norm(np.array(entry["loc"]) - np.array(loc)) < 1.5e3:
                        fout.write(line)

            match = next(filter(lambda x: entry["eid"].startswith(x), target_ts.keys()), None)

            if match is None:
                if global_radius > 0 and (entry["ts"] == 0 or abs(entry["ts"] - args.timestamp) < ts_eps):
                    if np.linalg.norm(np.array(entry["loc"]) - np.array(global_loc)) < global_radius:
                        fout.write(line)
                continue


            if match in eid_matches:
                if eid_matches[match] != entry["eid"]:
                    raise Exception(f'eid {match} is ambiguous. matches {eid_matches[match]} and {entry["eid"]}.')
            else:
                eid_matches[match] = entry["eid"]

            if abs(entry["ts"] - target_ts[match]) < ts_eps:
                if match in eid_ts:
                    raise Exception(f'already have ts {eid_ts[match]} for {match}, duplicate at {entry["ts"]}')
                eid_ts[match] = entry["ts"]
                fout.write(line)

        if len(eid_ts) != len(target_ts):
            missing = set(target_ts.keys()).difference(set(eid_ts.keys()))
            raise Exception(f'did not find matches for all eids, missing {len(missing)}: {missing}')
