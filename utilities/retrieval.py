class RetrievalConfig:
    def __init__(self, output, use_cache, use_dynamic_k, use_rerank):
        self.output = output
        self.use_cache = use_cache
        self.use_dynamic_k = use_dynamic_k
        self.use_rerank = use_rerank
        self.cache = {}