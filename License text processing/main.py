import cv2
import easyocr
import os
from zipfile import ZipFile
import numpy as np

def process_license(inputPath, outputPath):
    license_picture = cv2.imread(inputPath)

    ocr_reader = easyocr.Reader(['en'], gpu=False)
    result = ocr_reader.readtext(license_picture)

    for i in result:
        points = i[0]
        text = i[1]

        points = [(int(x), int(y)) for x, y in points]

        cv2.polylines(license_picture, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(license_picture, text, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imwrite(outputPath, license_picture)


if __name__ == "__main__":

    zippedFile = "./License Images.zip"
    extractedFolder = "./License Images"
    outputFolder = "./License Outputs"

    os.makedirs(extractedFolder)
    os.makedirs(outputFolder)

    with ZipFile(zippedFile, 'r') as zip_ref:
        zip_ref.extractall(extractedFolder)

    for fileName in os.listdir(extractedFolder):
        if fileName.endswith(".jpg"):
            inputPath = os.path.join(extractedFolder, fileName)
            outputPath = os.path.join(outputFolder, f"{fileName.split('.')[0]}_output.png")

            process_license(inputPath, outputPath)

    print("Processing completed.")