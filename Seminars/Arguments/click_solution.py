#!/usr/bin/env python

import click


@click.command()
@click.option("-o", "--output", help="The name of the output file", required=True)
@click.option("-r", "--rows", type=int, help="The number of rows to write", default=10)
@click.option(
    "-c", "--columns", type=int, help="The number of columns to write", default=10
)
@click.option(
    "-s", "--separator", help="The separator to use between values", default=","
)
def main(output, rows, columns, separator):
    print(f"output: {output}")
    print(f"rows: {rows}")
    print(f"columns: {columns}")
    print(f"separator: {separator}")


if __name__ == "__main__":
    main()
