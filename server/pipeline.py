from ml.fsl_svm_model import FslSvm

model = FslSvm("./csvs/train_filtered_v2.csv", "./csvs/test_filtered_v2.csv")

# # training model
# model.train_svm_model()
# model.save_svm_model()
# model.evaluate_svm_model()

# # test feed single video
# model.load_svm_model()
# model.run_svm_model("./clips/0/17.MOV")
#
# load model for evluation
model.load_svm_model(train_features="./data/train_features.npy", train_labels="./data/train_labels.npy", test_features="./data/test_features.npy", test_labels="./data/test_labels.npy")
model.evaluate_svm_model()

# # expand model to support more categories
# model.append_dataset(new_train_csv="./csvs/train_filtered_v3.csv", new_test_csv="./csvs/test_filtered_v3.csv")
# model.train_svm_model()
# model.save_svm_model()
# model.evaluate_svm_model()

# # load and train model
# model.load_Xy(train_features="./data/train_features_v3.npy", train_labels="./data/train_labels_v3.npy", test_features="./data/test_features_v3.npy", test_labels="./data/test_labels_v3.npy")
# model.train_svm_model()
# model.save_svm_model()
# model.evaluate_svm_model()
