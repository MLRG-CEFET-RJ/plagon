import numpy as np
import sys
import pickle
import argparse
import os
import random

def gen_examples(subdir, files):
    i = 0

    # see https://stackoverflow.com/questions/12668027/good-ways-to-expand-a-numpy-ndarray
    labels = np.array([])

    for triples_filename in files:
        infile = open(triples_filename, "rb")
        triples = pickle.load(infile)
        infile.close()

        expansion = np.zeros((len(triples),), dtype=int)
        labels = np.concatenate((labels,expansion), axis=0)

        for triple in triples:
            array1 = triple[0]
            array2 = triple[1]
            # see https://cmdlinetips.com/2018/04/how-to-concatenate-arrays-in-numpy/
            array_lst = np.concatenate((array1, array2))

            filename = os.path.join(subdir, 'ID{:010d}.npy'.format(i))
            np.save(filename, array_lst)
            labels[i] = triple[2]
            assert(labels[i] == 0 or labels[i] == 1)
            i = i + 1

    labels_filename = os.path.join(subdir, 'labels.npy')
    np.save(labels_filename, labels)

    return len(labels)

def main():
    parser = argparse.ArgumentParser(description='Generates examples to train the siamese net.')
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=.90, 
        required=False)
    parser.add_argument(
        "--val_ratio", 
        type=float, 
        default=.05, 
        required=False)
    parser.add_argument(
        "--triples_dir", 
        help = 'source directory of triples files', 
        required=True)
    parser.add_argument(
        "--destdir", 
        help = 'destination directory for the examples (one npy file per training example)', 
        required=True)
    parser.add_argument(
        "--max_docs", 
        type=int)
    args = parser.parse_args()

    filelist = os.listdir(args.triples_dir)
    filelist.sort()

    train_files = []
    val_files = []
    test_files = []
    
    n_train = 0
    n_val = 0
    n_test = 0

    num_docs = 0

    for file in filelist:
        filename = os.path.join(args.triples_dir, file)
        if filename.endswith('.pkl'):

            #print('Processing file %s' % filename)

            if num_docs > args.max_docs:
                break

            num_docs = num_docs + 1

            a_number = random.random()
            if a_number <= args.train_ratio:
                train_files.append(filename)
            elif a_number > args.train_ratio and a_number <= args.train_ratio + args.val_ratio:
                val_files.append(filename)
            else:
                test_files.append(filename)

    #print('Amounts of examples for training, validation and testing: %d, %d, %d' % (n_train, n_val, n_test))

    n_train = gen_examples(os.path.join(args.destdir, 'train'), train_files)
    print('Generated %d training examples.' % n_train)
    
    n_val = gen_examples(os.path.join(args.destdir, 'validation'), val_files)
    print('Generated %d validation examples.' % n_val)
    
    n_test = gen_examples(os.path.join(args.destdir, 'test'), test_files)
    print('Generated %d test examples.' % n_test)

'''
    Generate triples (examples) to be latter used to fit the the siamese neural network model.

    NB: a file of tuples may contain entries in which the similarity_bit is 0.5. These
        entries corresponds to combinations of sentences plag_x_plag. These entries are 
        not used to make training triples.

    Execution examples:
        python gen_examples.py --triples_dir /mnt/sdc/ebezerra/triples/ --destdir /mnt/sdc/ebezerra/datasets100 --max_docs 100
        python gen_examples.py --triples_dir /mnt/sdc/ebezerra/triples/ --destdir /mnt/sdc/ebezerra/datasets10 --max_docs 10
        python gen_examples.py --triples_dir /mnt/sdc/ebezerra/triples/ --destdir /mnt/sdc/ebezerra/datasets300 --max_docs 300
        python gen_examples.py --triples_dir ../data/triples/ --destdir ../data/datasets10 --max_docs 10
        ../../../../../data/home2/jrodrigues/datasets10/
'''
if __name__ == "__main__":
    main()
