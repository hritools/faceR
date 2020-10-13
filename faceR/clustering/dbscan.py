def plot_dist(images, embeddings):
    corr = euclidean_distances(embeddings, embeddings)

    # ~0.77 for 32fp, and ~21.7 for 16fp
    db = DBSCAN(eps=0.77, metric='euclidean', min_samples=4, n_jobs=-1).fit(embeddings)
    print(db)
    labels = db.labels_

    plt.imshow(corr, cmap='hot', interpolation='nearest')
    plt.show()

    pack = list(zip(labels, images))

    for t in pack:
        dir_name = 'faces/predicted/%d/' % t[0]
        filename = '%s/%s_%s' % (dir_name, t[1].split('/')[-2], t[1].split('/')[-1])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        copy2(t[1], filename)

    corr1 = corr.copy()
    # labels.sort()
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == -1 or labels[j] == -1:
                corr1[i][j] = 0.5
            elif labels[i] == labels[j]:
                corr1[i][j] = 0
            else:
                corr1[i][j] = 1
    plt.imshow(corr1, cmap='hot', interpolation='nearest')
    plt.show()

    c = -1
    prev = ''
    for i in range(len(images)):
        if images[i].split('/')[-2] != prev:
            c += 1
            prev = images[i].split('/')[-2]
        images[i] = images[i].split('/')[-2]

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(images, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(images, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(images, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(images, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(images, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(corr, labels))
