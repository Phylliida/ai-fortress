
import os
import dotenv
import base64
import aiohttp
import pathlib
import uuid
import json
from io import BytesIO
import math
from PIL import Image, PngImagePlugin, ImageChops, ImageDraw

RD_WIDTH = 32
RD_HEIGHT = 32
global GLOBAL_SEED
GLOBAL_SEED = 0

KCPP_STEPS = 11
KCPP_CFG_SCALE = 1
KCPP_WIDTH = 512
KCPP_HEIGHT = 512

def getRDApiKey():
    dotenv.load_dotenv()
    return os.getenv("RD_API_KEY")


async def makeRDRequest(payload):
    import requests
    
    headers = {
        "X-RD-Token": getRDApiKey(),
    }

    promptStyle = payload.get("prompt_style", "default")

    url = "https://api.retrodiffusion.ai/v1/inferences"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            responseJson = await resp.json()
    imagesBase64 = responseJson['base64_images']
    baseDirectory = pathlib.Path(f"./generated/rd/{promptStyle}")
    baseDirectory.mkdir(parents=True, exist_ok=True)
    paths = []
    for imageBase64 in imagesBase64:
        paths.append(base64ToFile(imageBase64, baseDirectory, payload))
    return paths

async def makeKoboldRequest(path, payload):
    url = f"http://localhost:5001/{path}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            responseJson = await resp.json()
    return responseJson


def base64ToFile(base64Str, baseDirectory, metadata):
    baseDirectory = pathlib.Path(str(baseDirectory))
    baseDirectory.mkdir(parents=True, exist_ok=True)
    imgPath = baseDirectory / (str(uuid.uuid4()) + ".png")
    imageBytes = base64.b64decode(base64Str)
    image = Image.open(BytesIO(imageBytes))
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", json.dumps(metadata))
    with open(imgPath, "wb") as f:
        image.save(f, format="PNG", pnginfo=info)
    return imgPath

def loadImage(imagePath):
    image = Image.open(imagePath)
    metadata = json.loads(image.info['prompt'])
    return image, metadata

def loadTileset(imagePath):
    image, metadata = loadImage(imagePath)

def pilToBase64(image: Image.Image, format: str = "PNG") -> str:
    """Return a base64-encoded string of the given PIL image."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")

async def makeSeamlessImage(prompt, negativePrompt, outputPath):

    payload = {
        "steps":KCPP_STEPS,
        "n":1,
        "sampler_name":"Euler",
        "width":KCPP_WIDTH,
        "height":KCPP_HEIGHT,
        "cfg_scale":KCCP_CFG_SCALE,
        "clip_skip":0,
        "seed":-1,
        "frames":1,
        "prompt":prompt,
        "negative_prompt":negativePrompt
    }

    result = await makeKoboldRequest("/sdapi/v1/txt2img", payload)
    for imageBase64 in result['images']:
        initialNonTilingImage = base64ToFile(imageBase64, "./generated/kcpp/txt2img", payload)
    print("Initial non tiling image", initialNonTilingImage)
    seamlessPath = await makeImageSeamless(prompt, negativePrompt, initialNonTilingImage, outputPath)



async def makeImageSeamless(prompt, negativePrompt, imagePath, outputPath):
    with Image.open(imagePath) as im:
        width, height = im.size
        shifted = ImageChops.offset(im, width // 2, height // 2)
        # Fill the exposed seams with wrapped pixels
        if im.mode == "RGBA":
            shifted = shifted.rotate(0)  # ensures alpha stays intact
    crossShape = generateCrossImage(width, height)

    payload = {
        "steps":11,
        "n":1,
        "sampler_name":"Euler",
        "width":width,
        "height":height,
        "cfg_scale":1,
        "clip_skip":0,
        "seed":-1,
        "denoising_strength":0.8,
        "frames":1,
        "prompt":prompt,
        "negative_prompt":negativePrompt,
        "init_images":[pilToBase64(shifted)],
        "inpainting_fill": 1,
        "inpainting_mask_invert": 0,
        "mask": pilToBase64(crossShape)
    }

    result = await makeKoboldRequest("/sdapi/v1/img2img", payload)
    for imageBase64 in result['images']:
        seamsRemovedPath = base64ToFile(imageBase64, "./generated/kcpp/steamfixup", payload)
    print("Seamless image step 1", seamsRemovedPath)
    with Image.open(seamsRemovedPath) as im:
        width, height = im.size
        shifted = ImageChops.offset(im, width // 2, height // 2)
        # Fill the exposed seams with wrapped pixels
        if im.mode == "RGBA":
            shifted = shifted.rotate(0)  # ensures alpha stays intact
    shifted.save("seamlessafter.png")
    return shifted



def generateCrossImage(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")

    minCrossStart = 0.33333
    maxCrossEnd = 0.666666
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)

    left = max(0, math.floor(width*minCrossStart))
    right = min(width, math.ceil(width*maxCrossEnd))
    top = max(0, math.floor(height*minCrossStart))
    bottom = min(height, math.ceil(height*maxCrossEnd))

    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)

    draw.rectangle([left, 0, right - 1, height - 1], fill="white")
    draw.rectangle([0, top, width - 1, bottom - 1], fill="white")

    return image


def offsetSeamless(imagePath, outputPath):
    srcPath = pathlib.Path(imagePath)
    if not srcPath.is_file():
        raise FileNotFoundError(f"Missing image: {srcPath}")

    with Image.open(srcPath) as im:
        width, height = im.size
        shifted = ImageChops.offset(im, width // 2, height // 2)
        # Fill the exposed seams with wrapped pixels
        if im.mode == "RGBA":
            shifted = shifted.rotate(0)  # ensures alpha stays intact
    
    shifted.save(pathlib.Path(outputPath))
    return shifted

async def makeTileset(prompt, extraPrompt=None):
    global GLOBAL_SEED
    promptStyle = "rd_tile__tileset" if extraPrompt is None else "rd_tile__tileset_advanced"
    payload = {
        "width": RD_WIDTH,
        "height": RD_HEIGHT,
        "prompt": prompt,
        "num_images": 1,
        "prompt_style": promptStyle,
        "seed": GLOBAL_SEED,
    }
    if extraPrompt is not None:
        payload['extra_prompt'] = extraPrompt
    return await makeRDRequest(payload)
