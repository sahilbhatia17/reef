import numpy as np
from data.loader import DataLoader
from sklearn import *
from lstm.imdb_lstm import *
from program_synthesis.heuristic_generator import HeuristicGenerator
from program_synthesis.synthesizer import Synthesizer
from program_synthesis.verifier import Verifier



import matplotlib.pyplot as plt

def run_user_study(userfilepath, userlabel):
    dl = DataLoader()
    dataset='imdb'
    train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
    train_ground, val_ground, test_ground, _, _, _ = dl.load_data_from_user(dataset='imdb', user_path=userfilepath)
    hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix, val_ground, train_ground, b=0.5)
    hg.run_synthesizer(max_cardinality=1, idx=None, keep=3, model='dt')
    syn = Synthesizer(val_primitive_matrix, val_ground, b=0.5)

    heuristics, feature_inputs = syn.generate_heuristics('nn', 1)
    print("Total Heuristics Generated: ", np.shape(heuristics)[1])
    top_idx = hg.prune_heuristics(heuristics, feature_inputs, keep=3)
    verifier = Verifier(hg.L_train, hg.L_val, val_ground, has_snorkel=False)

    verifier.train_gen_model()
    verifier.assign_marginals()

    feedback_idx = verifier.find_vague_points(gamma=0.1,b=0.5)
    print('Percentage of Low Confidence Points: ', np.shape(feedback_idx)[0]/float(np.shape(val_ground)[0]))
    validation_accuracy = []
    training_accuracy = []
    validation_coverage = []
    training_coverage = []

    training_marginals = []
    idx = None

    hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix, 
                                val_ground, train_ground, 
                                b=0.5)
    for i in range(3,26):
        if (i-2)%5 == 0:
            print("Running iteration: ", str(i-2))
            
        #Repeat synthesize-prune-verify at each iterations
        if i == 3:
            hg.run_synthesizer(max_cardinality=1, idx=idx, keep=3, model='dt')
        else:
            hg.run_synthesizer(max_cardinality=1, idx=idx, keep=1, model='dt')
        hg.run_verifier()
        
        #Save evaluation metrics
        va,ta, vc, tc = hg.evaluate()
        validation_accuracy.append(va)
        training_accuracy.append(ta)
        training_marginals.append(hg.vf.train_marginals)
        validation_coverage.append(vc)
        training_coverage.append(tc)
        
        #Find low confidence datapoints in the labeled set
        hg.find_feedback()
        idx = hg.feedback_idx
        
        #Stop the iterative process when no low confidence labels
        if idx == []:
            break
    filepath = './data/' + dataset
    # We save the training set labels Reef generates that we use in the next notebook to train a simple LSTM model.
    tlabelpath = filepath+'_reef_{}.npy'.format(userlabel)
    np.save(tlabelpath, training_marginals[-1])
    print("saved to {}".format(tlabelpath))

run_user_study("labeled_datapoints_jon")