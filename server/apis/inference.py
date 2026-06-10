from ml.fsl_svm_infer import FslSvmInfer

infer = FslSvmInfer("./models/fsl-svm-2-cat.pkl")


def process_frame(frame):
    try:
        pred = infer.predict(frame)

        if pred is None:
            return None

        return pred

    except Exception as e:
        return f"error: {str(e)}"
