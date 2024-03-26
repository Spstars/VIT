import timm
import torch
import numpy as np 


if __name__ == "__main__":

    # # 모델 이름과 pretrained 옵션을 지정합니다.
    # model_name = 'vit_base_patch16_224'  
    # pretrained = True 

    # timm 라이브러리를 사용하여 모델을 가져옵니다.
    # VisionTransformer에는 img_size를 조절할 수 있는 파라미터가 있다. 이걸 직접 가져와서 파라미터를 조절 할것임.
    # 기존 모델의 pretrained를 사용할 수 없는 것은 아쉬움.
    model= timm.models.VisionTransformer(img_size=(800,1024),patch_size=32,num_classes=10)
    input_size = (3, 800, 1024)  

    # 예시로 모델과 입력 크기를 출력합니다.
    print(model)
    print("Input Size:", input_size)