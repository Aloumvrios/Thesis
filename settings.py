max_datasets = 30
num_of_datasets = 30
num_of_important_features = 4
alpha = 0.05
kmer = 2
distance_threshold = 0.07  # [0.18, 0.21, 0.26, 0.3, 0.99] for method mean_cdist
method = 'permutation_pdist'  # permutation_pdist or KS_pdist or mean_cdist
fast_mode = False
fast_mode_str = 'fast' if fast_mode is True else 'slow'
no_of_classes = 3
fs = '_' + str(num_of_datasets) + '_' + \
              str(alpha) + '_' + \
              str(kmer) + '_' + \
              method + '_' + \
              str(no_of_classes)
file_suffix = fs + '_' + str(distance_threshold) if method == 'mean_cdist' else fs
