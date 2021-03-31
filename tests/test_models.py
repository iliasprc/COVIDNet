model = CovidNet('large', n_classes=3).cuda()

from torchsummary import summary

summary(model, (3, 224, 224), batch_size=8)
