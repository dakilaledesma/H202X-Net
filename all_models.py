import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)

print(model_names)