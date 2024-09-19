#!/usr/bin/env python

import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate Random CSV Data")
    parser.add_argument("-o", "--output", help="The name of the output file", required=True)
    parser.add_argument("-r", "--rows", type=int, help="The number of rows to write", default=10)
    parser.add_argument(
        "-c", "--columns", type=int, help="The number of columns to write", default=10
    )
    parser.add_argument(
        "-s", "--separator", help="The separator to use between values", default=","
    )
    args = parser.parse_args()
    output_file = args.output
    rows = args.rows
    columns = args.columns
    separator = args.separator
    print(f"output: {output_file}")
    print(f"rows: {rows}")
    print(f"columns: {columns}")
    print(f"separator: {separator}")


if __name__ == "__main__":
    main()
