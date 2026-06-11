from fsl_svm_model import FslSvm

model = FslSvm()
model.load_svm_model("svm_hog_model.pkl")

print(model.predict_frame(img_path="/home/code871/Downloads/imresizer-picture_2026-06-11_06-20-23(1).jpg"))


##for training the model
#model.train_svm_model()
#model.save_svm_model()
#model.evaluate_svm_model()

##for loading the model
#model.print_training_testing()
#model.load_svm_model()
#model.evaluate_svm_model()

