from typing import Optional
from vcf_data_loader import FixedSizeVCFChunks

class CacheGetTensor(object):
    def __init__(self, vcf: FixedSizeVCFChunks):
        self.vcf = vcf
        self._cache = None
        self._cached_chunk_index: Optional[int] = None

    def get_tensor_for_chunk_id(self, chunk: int):
        if chunk == self._cached_chunk_index:
            return self._cache

        self._cache = self.vcf.get_tensor_for_chunk_id(chunk)
        self._cached_chunk_index = chunk
        return self._cache


class Indexer(object):
    """Class for index arithmetic."""
    def __init__(self, vcf: FixedSizeVCFChunks, model_chunk_size: int):
        self.vcf = vcf
        self.cache = CacheGetTensor(vcf)
        self.model_chunk_size = model_chunk_size

        # Get VCF chunk size.
        cur = self.vcf.con.cursor()
        cur.execute("select count(*) from variants where chunk_id=0")
        self.vcf_chunk_size: int = cur.fetchone()[0]

    def get_chunk_index(self, idx: int):
        """Maps indices to sub chunks in the VCF.

        For example:
            If the model_chunk_size is 1k
            and the vcf chunk size is 5k
            get_chunk_index(6) is index 1000 in VCF chunk 1.

        """
        vcf_chunk_id = idx // self.vcf_chunk_size
        index_in_chunk = idx % self.vcf_chunk_size
        return (vcf_chunk_id, index_in_chunk)

    def extract_chunk(self, chunk_id: int, size=None):
        """Extract a chunk of model_chunk_size from the VCF.

        size can be used to override the chunk size returned. This is useful
        to get data for parent blocks that have a larger size.

        """
        idx = chunk_id * self.model_chunk_size
        vcf_chunk_id, index_in_chunk = self.get_chunk_index(idx)
        chunk = self.cache.get_tensor_for_chunk_id(vcf_chunk_id)

        if size is None:
            size = self.model_chunk_size

        return chunk[:, index_in_chunk:(index_in_chunk+size)]