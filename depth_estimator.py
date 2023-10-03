from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

# url = "/data/newhome/litianyi/dataset/EgoMotion/lab/02_01_walk/1/0028.jpg"
# image = Image.open(url)

# feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
# model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# # prepare image for the model
# inputs = feature_extractor(images=image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)
#     predicted_depth = outputs.predicted_depth

# # interpolate to original size
# prediction = torch.nn.functional.interpolate(
#     predicted_depth.unsqueeze(1),
#     size=image.size[::-1],
#     mode="bicubic",
#     align_corners=False,
# )

# # visualize the prediction
# output = prediction.squeeze().cpu().numpy()
# formatted = (output * 255 / np.max(output)).astype("uint8")
# depth = Image.fromarray(formatted)


from transformers import AutoImageProcessor, TimesformerForVideoClassification
import numpy as np
import torch
import torch.nn as nn
from model.loss import mpjpe
# video = torch.from_numpy(np.random.rand(8, 3, 224, 224))

# processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
# model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

# inputs = processor(video, return_tensors="pt")
# inputs['pixel_values'] = torch.repeat_interleave(inputs['pixel_values'], 3, dim=0)
# with torch.no_grad():
#   outputs = model(**inputs)
#   logits = outputs.logits


# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])

gt = torch.from_numpy(np.random.rand(32, 8, 17, 3))
predicted_3d_pos = torch.from_numpy(np.random.rand(32, 8, 17, 3))
error_1 = mpjpe(gt, predicted_3d_pos)
print("error_1 shape: ", error_1.shape)