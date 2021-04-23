import torch
from byol_pytorch import BYOL

from model import CNN, CovidNet, ViT

n_classes = 14

model = CovidNet('small', n_classes=n_classes).cuda()

# model = CovidNet('large', n_classes=n_classes)
#
# model = CNN(n_classes, 'mobilenet_v2')
#
# model = ViT(
#     image_size=224,
#     patch_size=32,
#     num_classes=3,
#     dim=512,
#     depth=6,
#     heads=16,
#     mlp_dim=1024,
#     dropout=0.1,
#     emb_dropout=0.1
# )
print(model)
learner = BYOL(
    model,
    image_size=224,
    hidden_layer='fc1'
)

opt = torch.optim.Adam(learner.parameters(), lr=1e-4)


def sample_unlabelled_images():
    return torch.randn(32, 3, 224, 224)


for i in range(200):
    images = sample_unlabelled_images()
    loss = learner(images.cuda())
    opt.zero_grad()
    loss.backward()
    if i%10==0:
        print(i,loss.item())
    opt.step()
    learner.update_moving_average()  # update moving average of target encoder

# save your improved network
cpkt = {}
cpkt['model_dict'] = model.state_dict()
torch.save(cpkt, './improved-covidnet_small.pt')
