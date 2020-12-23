import collections


def count_kmers(sequence, k):
    d = collections.defaultdict(int)
    for i in range(len(sequence)-(k-1)):
        d[sequence[i:i+k]] += 1
    return d


def probabilities(sequence, k):
    probability_dict = collections.defaultdict(float)
    N = len(sequence)
    kmer_count = count_kmers(sequence, k)
    for key, value in kmer_count.items():
        probability_dict[key] = float(value) / (N - k + 1)
    return probability_dict


def split_to_kmer(sequence, k):
    kmer_list = []
    for i in range(len(sequence) - (k - 1)):
        kmer_list.append(sequence[i:i + k])
    return kmer_list



