import faiss
import torch


class FaissOperator:
    @staticmethod
    def create_faiss_index(vectors, with_cuda=False):
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)

        if with_cuda:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)

        index.add(vectors)
        return index

    @staticmethod
    def search_similar_images(index, query_vector, k=5):
        distances, indices = index.search(query_vector, k)
        return indices

    @staticmethod
    def save_faiss_index(index, file_path):
        if faiss.get_num_gpus() > 0:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, file_path)

    @staticmethod
    def load_faiss_index(file_path, with_cuda=False):
        index = faiss.read_index(file_path)
        if with_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available.")
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
        return index
