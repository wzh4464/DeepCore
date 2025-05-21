class MetadataManager:
    """集中管理所有数据集元数据"""

    def __init__(self):
        self.metadata = {}

    def register_metadata(self, dataset_id, metadata_type, indices):
        """注册数据集元数据"""
        if dataset_id not in self.metadata:
            self.metadata[dataset_id] = {}
        self.metadata[dataset_id][metadata_type] = indices

    def get_metadata(self, dataset_id, metadata_type):
        """获取数据集元数据"""
        if dataset_id in self.metadata and metadata_type in self.metadata[dataset_id]:
            return self.metadata[dataset_id][metadata_type]
        return None
