def clusters_to_pairs(clusters, df):
    """Extract cross-source pairs from entity clusters for evaluation."""
    src = df.set_index("id")["source"]
    pairs = set()
    for cluster in clusters:
        members = list(cluster)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = str(members[i]), str(members[j])
                if src.get(a) != src.get(b):
                    pairs.add(tuple(sorted((a, b))))
    return pairs
