import base64
import io
from PIL import Image

def read_string():
    with open("hazard.jpg", "rb") as image:
        image_string = base64.b64encode(image.read())

    return image_string

def decode_base64():
    base64_string = read_string()
    decoded_string = io.BytesIO(base64.b64decode(base64_string))
    img = Image.open(decoded_string)
    return img.show()



if __name__ == "__main__":
    decode_base64()
