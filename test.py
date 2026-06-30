# import easyocr

# # Initialize the reader for English
# reader = easyocr.Reader(['en'])

# # Path to your handwritten image
# result = reader.readtext('drawing_preprocessed.jpg')

# # Print detected text
# for detection in result:
#     bbox, text, confidence = detection
#     print(f"Detected: '{text}' with confidence {confidence:.2f}")


# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image

# # Load model and processor
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# # Load image (replace path as needed)
# image = Image.open("drawing_preprocessed.jpg").convert("RGB")

# # Preprocess and predict
# pixel_values = processor(images=image, return_tensors="pt").pixel_values
# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print("Recognized:", generated_text)

import google.generativeai as genai
genai.configure(api_key="AIzaSyBwtbSe03fdtzJrnAZR-toITKFPvCfDsCc")
modeln="gemini-2.0-pro-exp"

uploaded_file = genai.Files.upload(file="drawing_preprocessed.jpg")
contents = [
    {
        "image": {
            "image_uri": uploaded_file.uri,  # or uploaded_file.name depending on response
        }
    },
    "ONLY REPLY WITH WHAT IS WRITTEN IN THE IMAGE. ONLY THE WORD THAT IS WRITTEN. NOTHING ELSE. NOTHING MORE."
]


response = model.generate_content(contents=contents)

print(response.text)

# from google import genai

# client = genai.Client(api_key="AIzaSyBwtbSe03fdtzJrnAZR-toITKFPvCfDsCc")

# my_file = client.files.upload(file="drawing_preprocessed.jpg")

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=[my_file, "ONLY REPLY WITH WHAT IS WRITTEN IN THE IMAGE. ONLY THE WORD THAT IS WRITTEN. NOTHING ELSE. NOTHING MORE."],
# )

# print(response.text)