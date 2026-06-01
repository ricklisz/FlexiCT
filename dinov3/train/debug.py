import argparse

import lmdb


def peek_keys(db, n=10):
    with lmdb.open(db, readonly=True, lock=False, subdir=False).begin() as txn:
        cur = txn.cursor()
        if cur.first():
            print(f"First {n} keys in {db}:")
            for _ in range(n):
                print("  ", cur.key().decode(errors="ignore"))
                if not cur.next():
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the first keys in an LMDB slice database.")
    parser.add_argument("db", help="Path to an LMDB file")
    parser.add_argument("--n", type=int, default=10, help="Number of keys to print")
    args = parser.parse_args()
    peek_keys(args.db, args.n)
