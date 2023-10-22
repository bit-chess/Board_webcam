from roboflow import Roboflow
rf = Roboflow(api_key="r3tT5JLqT1J3KY3T4BGA")
project = rf.workspace().project("chess-pieces-new")
model = project.version(19).model

# infer on a local image
print(model.predict("chess.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())