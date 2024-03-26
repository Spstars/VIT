import torch
import torch.nn.functional as F
import timm

def Vit_model_weight_load(image_size=(800,1024)):
    model_name = 'vit_base_patch16_224' 
    pretrained = True  # pretrained 옵션을 True로 설정하여 사전 학습된 가중치를 가져옵니다.
    # timm 라이브러리를 사용하여 모델을 가져옵니다.
    model = timm.create_model(model_name, pretrained=pretrained)
    # zero initialize feed_forward Network
    # with torch.no_grad():
    #     model.head.weight.zero_()
    #     model.head.bias.zero_()

    #linear interpolation
    # [1, 197, 768]
    print(model.patch_embed)  

    # print(dir(model))

    width, height = image_size
                                                    #50, 64
    target_patch_num_width ,target_patch_num_height = width//16, height//16
    # [1, 197, 768]
    pos_embed= model.pos_embed.detach()

    #remove positional embedding
     # [1, 196, 768]
    pos_embed = pos_embed[:,1:,:]
    idx_embed= pos_embed[:,0,:].reshape(1,1,768)
    pos_embed = pos_embed.reshape(1,-1,14,14)
    #our goal is to convert 16 x 16 to 50 * 64 with 2D interpolation, 
   # Perform 2D interpolation to resize positional embeddings
    resized_pos_embed = F.interpolate( pos_embed,
                                      size=(target_patch_num_height, target_patch_num_width), 
                                      mode='bicubic', align_corners=False).reshape(1,-1,768)
    model.pos_embed.data =torch.cat((idx_embed,resized_pos_embed),dim=1)
    model._process_input = None
    model.image_size = (800,1024)
    # print(model)
    return model
# Example usage:
if __name__ =="__main__":
    resized_pos_embed = Vit_model_weight_load(image_size=(800, 1024))

