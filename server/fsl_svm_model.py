from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, accuracy_score

class FslSvm:
    def __init__(self, training, testing):
        self.raw_train_data = pd.read_csv(training)
        self.raw_test_data = pd.read_csv(testing)
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.y_pred = None
        self.svm = None

    def _pre_processed_data(self, image_path: str):
        img = cv2.imread(image_path)

        #Normalize Image to 128 x 128
        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        #Convert to grayscale
        grayed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        #Otsu Thresholding
        _, thresh = cv2.threshold(
            grayed,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        #Morphological Operations
        kernel = np.ones((3,3), np.uint8)

        #Remove noise
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        #Fill gaps
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        #Hand Masking
        masked = cv2.bitwise_and(closing, closing, mask=closing)

        return masked

    def _processed_data(self, masked_img: np.ndarray):
        features, hog_image = hog(
            masked_img,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            visualize=True
        )
        
        return features

    def _build_Xy(self):
        if self.raw_train_data is None:
            print("404: No raw train data found")

            if self.raw_test_data is None:
                print("404: No raw test data found")

            return

        for row in tqdm(self.raw_train_data.itertuples(), total=len(self.raw_train_data)):
            pre_processed = self._pre_processed_data(row.filepath)
            features = self._processed_data(pre_processed)

            self.X_train.append(features)
            self.y_train.append(row.label)

        for row in tqdm(self.raw_test_data.itertuples(), total=len(self.raw_test_data)):
            pre_processed = self._pre_processed_data(row.filepath)
            features = self._processed_data(pre_processed)

            self.X_test.append(features)
            self.y_test.append(row.label)


        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        print("train shape:", self.X_train.shape)
        print("test shape:", self.X_test.shape)

    def _build_Xy_test(self):
        for row in tqdm(self.raw_test_data.itertuples(), total=len(self.raw_test_data)):
            pre_processed = self._pre_processed_data(row.filepath)
            features = self._processed_data(pre_processed)

            self.X_test.append(features)
            self.y_test.append(row.label)

        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

    def train_svm_model(self):
        if not self.X_train or not self.y_train or not self.X_test or not self.y_test:
            self._build_Xy()

        self.svm = SVC(
            kernel="linear",
            C=1,
            verbose=True,
            probability=True
        )

        self.svm.fit(self.X_train, self.y_train)

        print("SVM model are now trained.")

    def evaluate_svm_model(self):
        if self.svm is None:
            print("404: no trained svm model found.")

        if self.X_test is None or len(self.X_test) == 0:
            self._build_Xy_test()

        print("prediction test data in progress...")
        self.y_pred = self.svm.predict(self.X_test)

        #classification report
        print("classification report in progress...")
        print(classification_report(self.y_test, self.y_pred)) 

        #multiclass specificity
        print("multiclass specificity in progress...")
        mcm = multilabel_confusion_matrix(self.y_test, self.y_pred)
        class_labels = np.unique(self.y_test)

        results = []

        for label, cm_i in zip(class_labels, mcm):
            tn, fp, fn, tp = cm_i.ravel()
            specificity = tn / (tn + fp)

            results.append({
                "class": label,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "specificity": specificity
            })

        metrics_df = pd.DataFrame(results)

        print("Per-class Specificity Table:")
        print(metrics_df)

        #plot accuracy
        if self.X_train is not None and len(self.X_train) > 0:
            print("plot accuracy in progress...")
            train_acc = self.svm.score(self.X_train, self.y_train)
            test_acc = self.svm.score(self.X_test, self.y_test)

            print(f"Train Accuracy: {train_acc}")
            print(f"Test Accuracy: {test_acc}")

        #prediction results table
        print("prediction results in progress...")
        y_proba = self.svm.predict_proba(self.X_test)
        confidence = np.max(y_proba, axis=1)
        
        df = pd.DataFrame({
            "Index": range(len(self.y_test)),
            "True Label": self.y_test,
            "Predicted": self.y_pred,
            "Confidence": confidence
        })

        df["Correct"] = df["True Label"] == df["Predicted"]
        print(df.head())

        #confusion matrix
        print("confusion matrix in progress...")
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot()
        plt.title("SVM Confusion Matrix")
        plt.show()

    def load_svm_model(self, svm_model_path="/home/code871/Git/fsl-svm/server/svm_hog_model.pkl"):
        self.svm = joblib.load(svm_model_path)
        print("SVM model loaded.")

    def save_svm_model(self, svm_model_path="/home/code871/Git/fsl-svm/server/svm_hog_model.pkl"):

        if self.svm is None:
            print("404: no trained svm model found.")
            return

        joblib.dump(self.svm, svm_model_path)
        print("Model saved.")

    def print_training_testing(self):
        #print train data
        print(self.raw_train_data.head())
        print(self.raw_train_data.shape)

        #print test data
        print(self.raw_test_data.head())
        print(self.raw_test_data.shape)
