"""Command-line wrapper for converting old raw data to the new rating system."""

import argparse

from conversion import convert_file


parser = argparse.ArgumentParser(description="Convert old CSV rating points to the new top-30 club format.")
parser.add_argument("input_csv", help="Existing raw CSV with position/tournament_type/points columns.")
parser.add_argument("output_csv", help="Destination CSV with recalculated points.")
parser.add_argument("--drop-old-points", action="store_true", help="Do not keep the original points as old_points.")
args = parser.parse_args()

rows = convert_file(args.input_csv, args.output_csv, keep_old_points=not args.drop_old_points)
print(f"Converted {rows} rows to {args.output_csv}")