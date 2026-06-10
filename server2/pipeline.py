from fsl_svm_model import FslSvm

model = FslSvm()
model.load_svm_model("svm_hog_model.pkl")

print(model.predict_frame(img_path="/home/code871/Git/fsl-svm/server2/FSL-dataset/T/T_94.jpg"))


##for training the model
#model.train_svm_model()
#model.save_svm_model()
#model.evaluate_svm_model()

##for loading the model
#model.print_training_testing()
#model.load_svm_model()
#model.evaluate_svm_model()

