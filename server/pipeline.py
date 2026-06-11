from ml.fsl_svm_model import FslSvm

model = FslSvm("./csvs/train_filtered.csv", "./csvs/test_filtered.csv")

# # training model
# model.train_svm_model()
# model.save_svm_model()
# model.evaluate_svm_model()
#
# loading model
model.load_svm_model()
model.run_svm_model("/home/code871/Downloads/testMeGood.mov")
