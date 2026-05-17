from fsl_svm_model import FslSvm

model = FslSvm("/home/code871/Git/fsl-svm/server/csvs/train.csv", "/home/code871/Git/fsl-svm/server/csvs/test.csv")

##for training the model
#model.train_svm_model()
#model.save_svm_model()
#model.evaluate_svm_model()

#for loading the model
model.print_training_testing()
model.load_svm_model()
model.evaluate_svm_model()

