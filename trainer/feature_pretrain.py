model = CovidNet('small', n_classes=3).cuda()

from byol_pytorch import BYOL
from torchvision import models

resnet = models.resnet50(pretrained=True).cuda()

learner = BYOL(
    resnet,
    image_size = 224,
    hidden_layer = 'avgpool'
)

learner = BYOL(
    model,
    image_size = 224,
    hidden_layer = 'fc2'
)


opt = torch.optim.Adam(learner.parameters(), lr=1e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 224, 224).cuda()

for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    print(loss.item())
    opt.step()
    learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(model.state_dict(), '/content/drive/MyDrive/improved-covidnet_small.pt')