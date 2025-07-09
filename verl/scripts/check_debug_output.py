import torch

attention_mask = torch.load("/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/verl/_debug_output/debug_attention_mask_tensor.pt")
loss_mask = torch.load("/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/verl/_debug_output/debug_loss_mask_tensor.pt")
response_mask = torch.load("/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/verl/_debug_output/debug_response_mask_tensor.pt")

print("attention mask")
print(attention_mask)
print(attention_mask.shape)
print("attention mask mean")
print(attention_mask.float().mean())
# check if attention mask is different for each row
print("attention mask unique")
print(attention_mask.unique(dim=0).shape)

print("loss mask")
print(loss_mask)
print(loss_mask.shape)
print("loss mask mean")
print(loss_mask.float().mean())
print("loss mask unique")
print(loss_mask.unique(dim=0).shape)

print("response mask")
print(response_mask)
print(response_mask.shape)
print("response mask mean")
print(response_mask.float().mean())
print("response mask unique")
print(response_mask.unique(dim=0).shape)

response_tensor = torch.load("/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/verl/_debug_output/debug_responses_tensor.pt")
print("response tensor")
print(response_tensor[:, -2:])
print(response_tensor.shape)