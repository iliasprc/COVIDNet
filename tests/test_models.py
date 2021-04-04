from torchsummary import summary

from model import CNN, CovidNet, ViT

n_classes = 3

model = CovidNet('small', n_classes=n_classes)
summary(model, (3, 224, 224), batch_size=2,device='cpu')

model = CovidNet('large', n_classes=n_classes)
summary(model, (3, 224, 224), batch_size=2,device='cpu')

model = CNN(n_classes, 'mobilenet_v2')
summary(model, (3, 224, 224), batch_size=2,device='cpu')

model = ViT(
    image_size=224,
    patch_size=32,
    num_classes=3,
    dim=512,
    depth=6,
    heads=16,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)

summary(model, (3, 224, 224), batch_size=2,device='cpu')
