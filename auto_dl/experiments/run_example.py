from ingestion.data_ingestion import DataPreprocessing

a = DataPreprocessing()
t = a.get_train_test()

print(t)
