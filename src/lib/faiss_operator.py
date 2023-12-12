import faiss


class FaissOperator:
    @staticmethod
    def create_faiss_index(vectors):
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        return index

    @staticmethod
    def search_similar_images(index, query_vector, k=5):
        distances, indices = index.search(query_vector, k)
        return indices

    @staticmethod
    def save_faiss_index(index, file_path):
        faiss.write_index(index, file_path)

    @staticmethod
    def load_faiss_index(file_path):
        return faiss.read_index(file_path)
