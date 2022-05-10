import pickle
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_clusters(path_to_embeddings, plots, mode = 'random'):
	
	with open(path_to_embeddings,'rb') as pkl:
		corpus_embeddings =  pickle.load(pkl)
	num_clusters = 50
	clustering_model = KMeans(n_clusters=num_clusters)
	clustering_model.fit(corpus_embeddings)
	cluster_assignment = clustering_model.labels_
	cluster_centroids = clustering_model.cluster_centers_

	cluster_dict = {}
	cluster_embed = {}
	for idx,i in enumerate(cluster_assignment):
		if i not in cluster_dict:
			cluster_dict[i] = [plots[idx]]
			cluster_embed[i] = [corpus_embeddings[idx]]
		else:
			val = cluster_dict[i]
			val2 = cluster_embed[i]
			val.append(plots[idx])
			val2.append(corpus_embeddings[idx])
			cluster_dict[i] = val
			cluster_embed[i] = val2
	if mode in ['sorted-inc', 'sorted-dec', 'sorted-alt']:
		for k in cluster_embed.keys():
  			dist = cosine_similarity([cluster_centroids[k]],cluster_embed[k])
  			keys = np.argsort(dist)
  			lst = np.array(cluster_dict[k])[keys]
  			cluster_dict[k] = list(lst[0])
	#naive strategy - choose k distinct cluster in each iteration
	cluster_data = []
	clusters_complete = []
	cluster_choice = []
	done = True
	batch_size = 5
	bool_dir = np.ones(num_clusters)
	while done:
		if len(clusters_complete) < num_clusters - batch_size:
			choice = np.random.choice(np.delete(range(num_clusters), clusters_complete), batch_size, replace=False)
		else:
			choice = np.random.choice(np.delete(range(num_clusters), clusters_complete), num_clusters-len(clusters_complete), replace=False)
		for k in choice:
			if len(cluster_dict[k]) > 0:
				if mode == "random" or 'sorted-inc' :
					cluster_data.append(cluster_dict[k].pop(0))
				elif mode == 'sorted-dec':
					cluster_data.append(cluster_dict[k].pop(-1))
				elif mode == 'sorted-alt':
					if 	bool_dir[k] == 1:
						cluster_data.append(cluster_dict[k].pop(0))
						bool_dir[k] == 0
					else:
						cluster_data.append(cluster_dict[k].pop(-1))
						bool_dir[k] == 1
				cluster_choice.append(k)
			else:
				clusters_complete.append(k)
				clusters_complete = list(set(clusters_complete))
				if len(clusters_complete) == len(cluster_dict.keys()):
					done = False
	with open('clustering_data.pickle','wb') as myfile:
		pickle.dump([cluster_data,cluster_choice], myfile)
	return cluster_data, cluster_choice
