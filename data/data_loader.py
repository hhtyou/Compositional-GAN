
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    #print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
def CreateLandmarkDataLoader(opt):
    from data.custom_dataset_data_loader import CustomMarkDatasetDataLoader
    landmark_data_loader = CustomMarkDatasetDataLoader()
    landmark_data_loader.initialize(opt)
    return landmark_data_loader
