import timm
from pprint import pprint
model_names = timm.list_models()

print('\n'.join(model_names))
