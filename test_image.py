import os
import base64

from together import Together
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import traceback

load_dotenv()

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

response = client.images.generate(
    prompt="A beautiful sunset over a mountain range",
    model="black-forest-labs/FLUX.1-schnell-Free",
    width=1024,
    height=768,
    steps=4,
    n=1,
    response_format="b64_json",
)

b64_image = response.data[0].b64_json

image_data = base64.b64decode(b64_image)

with open("generated_image.png", "wb") as f:
    f.write(image_data)

print("Image saved as 'generated_image.png'")
