def search_similar_images(query_features, index, image_paths, class_names, k=5):
    distances, indices = index.search(query_features, k)
    similar_images = []
    predicted_classes = []

    for j, i in enumerate(indices[0]):
        similar_images.append((image_paths[i], distances[0][j]))
        predicted_classes.append(class_names[i])

    return similar_images, predicted_classes
