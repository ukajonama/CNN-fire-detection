import cv2
import numpy as np
from keras.models import load_model
import os
# 저장된 모델 로드
model = load_model('model.h5')
# 이미지 폴더 경로
test_folder = r'D:\github\CNN_fire_Detection\test'
# 결과 저장용 파일
result_file = r'D:\github\CNN_fire_Detection\results.txt'
# 크기 맞추기
IMG_SIZE = 100
# result_file을 작성한다.
with open(result_file, 'w') as f:
    for img in os.listdir(test_folder):
        try:
            # 이미지 로드 및 크기 변경
            img_array = cv2.imread(os.path.join(test_folder, img))
            resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            input_data = np.expand_dims(resized_img_array, axis=0).astype('float32') / 255.0
            
            # 이미지를 모델에 전달하고 결과 예측
            prediction = model.predict(input_data)
            if prediction[0][0] < 0.5:
                fire_status = "fire"
            else:
                fire_status = "no_fire"
            
            # 이미지 파일 이름과 결과 출력 및 저장
            print(f"{img}: {fire_status}")
            f.write(f"{img}: {fire_status}\n")
        except Exception as e:
            print(f"Error: {e}")
