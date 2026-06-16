import hog
import os

hogResult = []

for imagePath in os.listdir("cifar_bw/"):
    hogResult.append(hog.hog_from_path("cifar_bw/"+imagePath).tolist())
    #print(hogResult)

hogFile = open("expected.txt", "w")
resultStr = ""

for hogFeature in hogResult:
    resultStr += str(hogFeature) + "\n"

hogFile.write(resultStr)
hogFile.close()