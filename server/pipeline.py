from ml.fsl_svm_model import FslSvm

model = FslSvm("./csvs/train_filtered.csv", "./csvs/test_filtered.csv")

# training model
# model.train_svm_model()
# model.save_svm_model()
# model.evaluate_svm_model()


# test feed single video
model.load_svm_model()
model.run_svm_model("/home/code871/Git/fsl-svm/server/clips/1/20.MOV")
#
# # load model for evluation
# model.load_svm_model(train_features="./data/train_features.npy", train_labels="./data/train_labels.npy", test_features="./data/test_features.npy", test_labels="./data/test_labels.npy")
# model.evaluate_svm_model()

