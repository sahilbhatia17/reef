import pickle
from sklearn.cluster import KMeans

def generate_clusters(path_to_embeddings, plots):
	with open(path_to_embeddings,'rb') as pkl:
		corpus_embeddings =  pickle.load(pkl)
	num_clusters = 8
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

	#naive strategy - alternate data between different clusters
	cluster_data = []
	count = 0
	clusters_complete = []
	done = True
	while done:
		for keys in cluster_dict.keys():
			if count < len(cluster_dict[keys]):
				cluster_data.append(cluster_dict[keys][count])
			else:
				clusters_complete.append(keys)
				clusters_complete = list(set(clusters_complete))
				if len(clusters_complete) == len(cluster_dict.keys()):
					done = False
		count += 1
	return cluster_data
