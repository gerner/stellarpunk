""" Tool to grab relevant entries from a debug history """

import sys
import argparse
import json
import gzip
import contextlib
import logging

from stellarpunk import util

TS_EPS = 1/60/2

def main():
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
        parser.add_argument("--pdb", action="store_true")

        args = parser.parse_args()

        if args.pdb:
            context_stack.enter_context(util.PDBManager())

        target_ts = {}
        for pattern in args.eid:
            eid, _, ts = pattern.partition(":")
            if ts != "":
                target_ts[eid] = float(ts)
            else:
                target_ts[eid] = args.timestamp

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

        eid_matches = {}
        eid_ts = {}
        for line in fin:
            entry = json.loads(line)
            match = next(filter(lambda x: entry["eid"].startswith(x), target_ts.keys()), None)

            if match is None:
                continue

            if match in eid_matches:
                if eid_matches[match] != entry["eid"]:
                    raise Exception(f'eid {match} is ambiguous. matches {eid_matches[match]} and {entry["eid"]}.')
            else:
                eid_matches[match] = entry["eid"]

            if abs(entry["ts"] - target_ts[match]) < TS_EPS:
                eid_ts[match] = entry["ts"]
                fout.write(line)

        if len(eid_ts) != len(target_ts):
            missing = set(target_ts.keys()).difference(set(eid_ts.keys()))
            raise Exception(f'did not find matches for all eids, missing {len(missing)}: {missing}')
