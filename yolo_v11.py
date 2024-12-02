from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11x.pt")

    # # Train the model
    # train_results = model.train(
    #     data="coco8.yaml",  # path to dataset YAML
    #     epochs=1,  # number of training epochs
    #     imgsz=640,  # training image size
    #     device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    # )

    # Evaluate model performance on the validation set
    # metrics = model.val(data="coco8.yaml")

    # Perform object detection on an image
    results = model("2024_NIA_CM_SOU_REAL_M_A2_OA_NA_D_03324_Pre.JPG")
    print(results.shape())
    # results[1].show()

    # # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model