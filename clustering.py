import pickle
from sklearn.cluster import KMeans
import numpy as np

def generate_clusters(path_to_embeddings, plots):
	with open(path_to_embeddings,'rb') as pkl:
		corpus_embeddings =  pickle.load(pkl)
	num_clusters = 50
	clustering_model = KMeans(n_clusters=num_clusters)
	clustering_model.fit(corpus_embeddings)
	cluster_assignment = clustering_model.labels_

	cluster_dict = {}
	for idx,i in enumerate(cluster_assignment):
		if i not in cluster_dict:
			cluster_dict[i] = [plots[idx]]
		else:
			val = cluster_dict[i]
			val.append(plots[idx])
			cluster_dict[i] = val

	#naive strategy - choose k distinct cluster in each iteration
	cluster_data = []
	clusters_complete = []
	cluster_choice = []
	done = True
	batch_size = 5
	while done:
		if len(clusters_complete) < num_clusters - batch_size:
			choice = np.random.choice(np.delete(range(num_clusters), clusters_complete), batch_size, replace=False)
		else:
			choice = np.random.choice(np.delete(range(num_clusters), clusters_complete), num_clusters-len(clusters_complete), replace=False)
		for k in choice:
			if len(cluster_dict[k]) > 0:
				cluster_data.append(cluster_dict[k].pop(0))
				cluster_choice.append(k)
			else:
				clusters_complete.append(k)
				clusters_complete = list(set(clusters_complete))
				if len(clusters_complete) == len(cluster_dict.keys()):
					done = False
	return cluster_data, cluster_choice
