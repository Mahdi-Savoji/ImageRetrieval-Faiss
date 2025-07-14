def search_similar_images(query_features, index, image_paths, k=5):
    distances, indices = index.search(query_features, k)
    similar_images = [(image_paths[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return similar_images