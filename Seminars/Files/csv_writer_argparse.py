#!/usr/bin/env python

import argparse
import random


def random_data() -> int:
    return random.randint(0, 100)


def main():
    parser = argparse.ArgumentParser(description="Generate Random CSV Data")
    parser.add_argument(
        "-o", "--output", help="The name of the output file", required=True
    )
    parser.add_argument(
        "-r", "--rows", type=int, help="The number of rows to write", default=10
    )
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

    with open(output_file, "w") as file:
        for row in range(rows):
            for column in range(columns):
                file.write(f"{random_data()}")
                if column < columns - 1:
                    file.write(separator)
            file.write("\n")


if __name__ == "__main__":
    main()
