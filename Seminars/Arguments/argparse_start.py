#!/usr/bin/env python

import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate Random CSV Data")
    parser.add_argument("-o", "--output", help="The name of the output file")
    parser.add_argument("-r", "--rows", type=int, help="The number of rows to write")
    parser.add_argument(
        "-c", "--columns", type=int, help="The number of columns to write"
    )
    parser.add_argument("-s", "--separator", help="The separator to use between values")
    args = parser.parse_args()
    print(args)
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
