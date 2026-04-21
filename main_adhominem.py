# -*- coding: utf-8 -*-
from adhominem import AdHominem
import pickle
import os
import argparse

def load_adhominem_tsv(path):
    docs_L, docs_R, labels = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(f"[BAD LINE {i}] no tab found:")
                print(line[:300])
                continue

            label, text = parts

            if "$$$" not in text:
                print(f"[BAD LINE {i}] no $$$ found:")
                print(text[:300])
                continue

            text1, text2 = text.split("$$$", 1)

            docs_L.append(text1)
            docs_R.append(text2)
            labels.append(int(label))

    print("Loaded pairs:", len(labels))
    return docs_L, docs_R, labels

def main():

    parser = argparse.ArgumentParser(description='AdHominem - Siamese Network for Authorship Verification')
    parser.add_argument('-D_c', default=10, type=int)  # character embedding dimension
    parser.add_argument('-D_r', default=30, type=int)  # character representation dimension
    parser.add_argument('-w', default=4, type=int)  # length of 1D-CNN sliding window
    parser.add_argument('-D_w', default=300, type=int)  # dimension of word embeddings
    parser.add_argument('-D_s', default=50, type=int)  # dimension sentence embeddings
    parser.add_argument('-D_d', default=50, type=int)  # dimension of document embedding
    parser.add_argument('-D_mlp', default=60, type=int)  # final output dimension
    parser.add_argument('-T_c', default=15, type=int)  # maximum number of characters per words
    parser.add_argument('-T_w', default=20, type=int)  # maximum number of words per sentence
    parser.add_argument('-T_s', default=50, type=int)  # maximum number of sentences per document
    parser.add_argument('-t_s', default=0.95, type=float)  # boundary for similar pairs (close to one)
    parser.add_argument('-t_d', default=0.05, type=float)  # boundary for dissimilar pairs (close to zero)
    parser.add_argument('-epochs', default=100, type=int)  # total number of epochs
    parser.add_argument('-train_word_embeddings', default=False, type=bool)  # fine-tune pre-trained word embeddings
    parser.add_argument('-batch_size_tr', default=32, type=int)  # batch size for training
    parser.add_argument('-batch_size_te', default=128, type=int)  # batch size for evaluation
    parser.add_argument('-initial_learning_rate', default=0.002, type=float)  # initial learning rate
    parser.add_argument('-keep_prob_cnn', default=0.9, type=float)  # dropout for 1D-CNN layer
    parser.add_argument('-keep_prob_lstm', default=0.9, type=float)  # variational dropout for BiLSTM layer
    parser.add_argument('-keep_prob_att', default=0.9, type=float)  # dropout for attention layer
    parser.add_argument('-keep_prob_metric', default=0.9, type=float)  # dropout for metric learning layer

    parser.add_argument("-mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("-ckpt", type=str, default="checkpoints/adhominem")
    args = parser.parse_args()

    hyper_parameters = vars(parser.parse_args())

    # create folder for results
    dir_results = os.path.join('results')
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    # load docs, vocabularies and initialized word embeddings
    with open(os.path.join("data", "data_PAN20train_small"), 'rb') as f:
        docs_L_tr, docs_R_tr, labels_tr, \
        docs_L_te, docs_R_te, labels_te, \
        V_w, E_w, V_c = pickle.load(f)

    # add vocabularies to dictionary
    hyper_parameters['V_w'] = V_w
    hyper_parameters['V_c'] = V_c
    hyper_parameters['N_tr'] = len(labels_tr)
    hyper_parameters['N_dev'] = len(labels_te)

    # file to store results epoch-wise
    file_results = os.path.join(dir_results, 'results.txt')

    # delete already existing files
    if os.path.isfile(file_results):
        os.remove(file_results)

    # write hyper-parameters setup into file (results.txt)
    open(file_results, 'a').write('\n'
                                   + '--------------------------------------------------------------------------------'
                                   + '\nPARAMETER SETUP:\n'
                                   + '--------------------------------------------------------------------------------'
                                   + '\n'
                                   )
    for hp in hyper_parameters.keys():
        if hp in ['V_c', 'V_w']:
            open(file_results, 'a').write('num ' + hp + ': ' + str(len(hyper_parameters[hp])) + '\n')
        else:
            open(file_results, 'a').write(hp + ': ' + str(hyper_parameters[hp]) + '\n')

    # load neural network model
    adhominem = AdHominem(hyper_parameters=hyper_parameters,
                          E_w_init=E_w,
                          )
    # start training
    train_set = (docs_L_tr, docs_R_tr, labels_tr)
    test_set = (docs_L_te, docs_R_te, labels_te)
    

    if args.mode == "train":
        adhominem.train_model(train_set, test_set, file_results)

    elif args.mode == "test":
        adhominem.restore_model(args.ckpt)

        test_file = os.environ.get("ADH_AMAZON_TEST_FILE", None)
        if test_file:

            split_name = os.path.splitext(os.path.basename(test_file))[0]
            docs_L_te, docs_R_te, labels_te = load_adhominem_tsv(test_file)
            print("Using external test file:", test_file)
        else:
            split_name = "PAN20_fixed_test"
            print("Using default PAN20 test set from pickle")

        adhominem.evaluate_model(
            docs_L_te,
            docs_R_te,
            labels_te,
            batch_size=40,
            split_name=split_name
        )
    # close session
    adhominem.sess.close()


if __name__ == '__main__':
    main()
