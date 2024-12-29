import os
import base64
import logging
import asyncio
import webbrowser
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from together import Together
import traceback
import os
import tweepy
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

def optimize_tweet(tweet: str) -> str:
    """
    Améliore un tweet pour le rendre plus professionnel/engageant.
    """
    prompt = (
        "Vous êtes un expert en réseaux sociaux. Prenez le texte suivant, améliorez-le pour qu'il soit engageant, "
        "professionnel et ajoutez des hashtags pertinents. Ne changez pas le contexte ou le message principal :\n\n"
        f"Tweet original : {tweet}\n\n"
        "Tweet optimisé :"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    response = llm.invoke([
        ("system", "Vous êtes un assistant expert en réseaux sociaux."),
        ("human", prompt)
    ])
    return response.content.strip()


def tweet_posting_tool(tweet: str) -> str:
    """
    Publie un tweet via l'API Twitter.

    Param:
      tweet (str): Le texte du tweet à publier.
    Retour:
      str: Un message de réussite ou d'erreur.
    """
    consumer_key = os.getenv("TWITTER_API_KEY")
    consumer_secret = os.getenv("TWITTER_API_SECRET_KEY")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

    if not all([consumer_key, consumer_secret, access_token, access_secret]):
        return "Erreur : Clés Twitter absentes ou incomplètes dans l'environnement."

    try:
        client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_secret
        )
        client.create_tweet(text=optimize_tweet(tweet))
        return f"Tweet publié avec succès : {tweet}"
    except Exception as e:
        return f"Échec de la publication. Erreur : {str(e)}"
    



load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

scratch_pad_dir = "../scratchpad"
os.makedirs(scratch_pad_dir, exist_ok=True)

# Client Together
together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

class EnhancedPrompt(BaseModel):
    """
    Classe pour structurer le prompt enrichi
    """
    content: str = Field(
        ...,
        description="Prompt enrichi pour générer une image",
    )


async def generate_image_handler(prompt: str):
    """
    Génère une image basée sur un prompt donné en utilisant l'API Together.
    """
    try:
        logger.info(f"✨ Enrichissement du prompt : '{prompt}'")

        llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.25,
            max_retries=2,
        )

        structured_llm = llm.with_structured_output(EnhancedPrompt)

        system_template = """
        Enhance the given prompt with the best prompt engineering techniques, such as providing context, specifying style, medium, lighting, and camera details if applicable. If the prompt requests a realistic style, the enhanced prompt should include the image extension .HEIC.

        # Original Prompt
        {prompt}

        # Objective
        **Enhanced Prompt**: Add relevant details to the prompt, including context, description, specific visual elements, mood, and technical details. For realistic prompts, add '.HEIC' in the output specification.

        # Example
        "realistic photo of a person having a coffee" -> "photo of a person having a coffee in a cozy cafe, natural morning light, shot with a 50mm f/1.8 lens, 8425.HEIC"
        """

        prompt_template = PromptTemplate(
            input_variables=["prompt"],
            template=system_template,
        )

        chain = prompt_template | structured_llm

        enhanced_prompt = chain.invoke({"prompt": prompt}).content

        logger.info(f"🌄 Génération d'image basée sur le prompt enrichi : '{enhanced_prompt}'")
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

        logger.info(f"🖼️ Image générée et sauvegardée avec succès à {img_path}")

        return {
            "message": f"Image générée avec le prompt enrichi : '{enhanced_prompt}'",
            "image_path": img_path,
        }

    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération de l'image : {str(e)}\n{traceback.format_exc()}")
        return {"error": str(e)}
