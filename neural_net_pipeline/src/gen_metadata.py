import numpy as np
import sys
from scipy import spatial
import sqlite3 as lite
import pickle
from itertools import islice
from pan_db import PanDatabaseManager
import argparse

def load_metadata(path):
    dbfile = open(path, 'rb')
    metadata = pickle.load(dbfile)
    return metadata

def main():
    parser = argparse.ArgumentParser(
        description="Arguments to complete metadata"
    )
    parser.add_argument(
        "--pandb",
        help="specify the sqlite PAN database file containing the input sentences",
        required=True,
    )
    parser.add_argument(
        "--destdir",
        help="destination directory for the metadata to be generated",
        required=True,
    )

    args = parser.parse_args()

    pandb = PanDatabaseManager(args.pandb)
    sentences_ids, article_ids, author_ids, isplag_flags = pandb.get_sentences_metadata()

    print("Amount of sentence ids: ", len(sentences_ids))

    #Saving metadata to pickle file...
    dbfile = open(args.destdir, 'wb')
    metadata_dict = {'sentences_ids': sentences_ids, 'article_ids': article_ids, 'author_ids': author_ids, 'isplag_flags': isplag_flags}
    pickle.dump(metadata_dict, dbfile)
    dbfile.close()

    print("Done!")

    metadata = load_metadata(args.destdir)
    article_ids = metadata['article_ids']
    print(article_ids[:200])

    metadata = load_metadata()
    sentences_ids = metadata['sentences_ids']
    print(sentences_ids[:10])

"""
    Execution example:
        python gen_metadata.py --pandb '../data/pan_db' --destdir '../data/pkl/metadata.pkl'
"""
if __name__ == "__main__":
    main()
