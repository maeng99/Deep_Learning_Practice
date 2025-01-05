import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# 모델 불러오기
model = ResNet101(weights='imagenet')

# 이미지 파일 불러오기
img_path = 'images/cat_224x224.jpg'  # 이미지 경로 설정
img = image.load_img(img_path, target_size=(224, 224))  # 모델에 맞는 이미지 크기로 조정

# 이미지를 배열로 변환하고 전처리
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 이미지에 대해 예측
predictions = model.predict(x)

# 예측 결과 해석
decoded_predictions = decode_predictions(predictions, top=3)[0]
print('Predictions:')
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")