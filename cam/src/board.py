import cv2
from roboflow import Roboflow

rf = Roboflow(api_key="r3tT5JLqT1J3KY3T4BGA")
project = rf.workspace().project("chess-pieces-wrdbb")
model = project.version(3).model

cam = cv2.VideoCapture(0)

while cam.isOpened():
    success, image = cam.read()

    previsoes = model.predict(image, confidence=40, overlap=30)

    for previsao in previsoes:
        if previsao is not None:
            x = previsao["x"] - 20
            y = previsao["y"] - 20
            w = previsao["width"]
            h = previsao["height"]
            confianca = previsao["confidence"]  # Confiança da previsão
            classe = previsao["class"]  # Nome da classe da previsão

            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)  # Cor verde, espessura 2
            cv2.putText(image, f"{classe}: {confianca:.2f}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Output", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
