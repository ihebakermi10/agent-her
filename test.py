"""
Derived from https://github.com/Chainlit/cookbook/tree/main/realtime-assistant
and https://github.com/disler/poc-realtime-ai-assistant/tree/main
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess
import base64
import os
import logging
import webbrowser

from together import Together
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import traceback

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Scratchpad directory
scratch_pad_dir = "../scratchpad"
os.makedirs(scratch_pad_dir, exist_ok=True)

# Initialize clients
together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# Define the tools
class EnhancedPrompt(BaseModel):
    """
    Class for the text prompt
    """

    content: str = Field(
        ...,
        description="The enhanced text prompt to generate an image",
    )


async def generate_image_handler(prompt):
    """
    Generates an image based on a given prompt using the Together API.
    """
    try:

        logger.info(f"✨ Enhancing prompt: '{prompt}'")

        llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.25,
            max_retries=2,
        )

        structured_llm = llm.with_structured_output(EnhancedPrompt)

        system_template = """
        Enhance the given prompt the best prompt engineering techniques such as providing context, specifying style, medium, lighting, and camera details if applicable. If the prompt requests a realistic style, the enhanced prompt should include the image extension .HEIC.

        # Original Prompt
        {prompt}

        # Objective
        **Enhance Prompt**: Add relevant details to the prompt, including context, description, specific visual elements, mood, and technical details. For realistic prompts, add '.HEIC' in the output specification.

        # Example
        "realistic photo of a person having a coffee" -> "photo of a person having a coffee in a cozy cafe, natural morning light, shot with a 50mm f/1.8 lens, 8425.HEIC"
        """

        prompt_template = PromptTemplate(
            input_variables=["prompt"],
            template=system_template,
        )

        chain = prompt_template | structured_llm

        # Generate the enhanced prompt
        enhanced_prompt = chain.invoke({"prompt": prompt}).content

        logger.info(f"🌄 Generating image based on prompt: '{enhanced_prompt}'")
        response = together_client.images.generate(
            prompt=enhanced_prompt,  
            model="black-forest-labs/FLUX.1-schnell-Free",
            width=1024,
            height=768,
            steps=4,
            n=1,
            response_format="b64_json",
        )

        b64_image = response.data[0].b64_json
        image_data = base64.b64decode(b64_image)

        img_path = os.path.join(scratch_pad_dir, "generated_image.jpeg")
        with open(img_path, "wb") as f:
            f.write(image_data)

        logger.info(f"🖼️ Image generated and saved successfully at {img_path}")
        

        return {
            "message": f"Image generated with the prompt '{enhanced_prompt}'",
            "image_path": img_path
        }

    except Exception as e:
        logger.error(f"❌ Error generating image: {str(e)}\n{traceback.format_exc()}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Exemple de prompt pour créer un avatar
    sample_prompt = "realistic avatar of a young woman with blue eyes and long hair"

    async def main():
        result = await generate_image_handler(sample_prompt)
        if "error" in result:
            logger.error(f"Test failed: {result['error']}")
        else:
            logger.info(result["message"])
            logger.info(f"Image saved at: {result['image_path']}")
            webbrowser.open(f"file://{os.path.abspath(result['image_path'])}")

    # Exécuter la fonction asynchrone
    asyncio.run(main())
