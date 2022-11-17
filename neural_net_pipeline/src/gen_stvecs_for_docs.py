import sys, os
from sentence_encoder import UniversalSentenceEncoder

#sys.path.append("skip-thoughts.torch/pytorch")

import argparse
from pan_db import PanDatabaseManager
import pickle
import tensorflow as tf
import tensorflow_hub as hub

MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4" 
print('________________________________________________________________________')
LOADED_MODEL = hub.load(MODULE_URL)
print ("module %s loaded" % MODULE_URL)
print('________________________________________________________________________')

# Print iterations progress
# source: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def main():
    BATCH_SIZE = 512

    parser = argparse.ArgumentParser(
        description="Generates SkipThouth vectors for sentences from the PAN corpus."
    )
    parser.add_argument(
        "--pandb",
        help="specify the sqlite PAN database file containing the input sentences",
        required=True,
    )
    parser.add_argument(
        "--debug", action="store_true", help="print debug messages to stderr"
    )
    parser.add_argument(
        "--destdir",
        help="destination directory for the sentence vectors to be generated",
        required=True,
    )
    parser.add_argument(
        "--start",
        type=int,
        help="doc id at which the processing should start",
        required=True,
    )

    args = parser.parse_args()

    pandb = PanDatabaseManager(args.pandb)
    sentences = pandb.get_sentence_texts()
    print("Amount of sentences to be processed: ", len(sentences))

    ids_for_docs = pandb.get_ids_for_documents()

    # printProgressBar(
    #     0, len(ids_for_docs), prefix="Progress:", suffix="Complete", length=50
    # )
    for doc_id in ids_for_docs:

        if doc_id < args.start:
            continue

        sentences = pandb.get_sentences_texts_for_doc(doc_id)
        l = len(sentences)
        array = []

        # possível adaptação ##########################
        if l > 3000:
            print(
                "Skipping doc %d because it is too large (%d sentences)." % (doc_id, l)
            )
            array.append("tuples{:05d}.pkl".format(doc_id))
            continue
        
        sentence_encoder = UniversalSentenceEncoder(LOADED_MODEL)
        vec_list = []
        for sentence in sentences:
            vec = sentence_encoder.encode_sentence(sentence)
            vec_list.append(vec)
        # Saving batch of sentence embeddings to pickle file...
        pkl_filename = os.path.join(args.destdir, "stvecs{:05d}.pkl".format(doc_id))

        print("Dumping %d sentence vectors for doc %d." % (l, doc_id))
        dbfile = open(pkl_filename, "wb")
        
        pickle.dump(vec_list, dbfile)
        dbfile.close()

        # printProgressBar(
        #     doc_id, len(ids_for_docs), prefix="Progress:", suffix="Complete", length=50
        # )

    print(array)
    print("Done!")


"""
    Execution example:
        python gen_stvecs_for_docs.py --pandb '../data/pan_db' --destdir '../data/stvecs' --start 1
"""
if __name__ == "__main__":
    main()
