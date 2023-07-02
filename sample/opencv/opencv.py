import numpy as np
import requests
import cv2
from cairosvg import svg2png
from PIL import image

def svgToPng(svg_url):
    svg_data = requests.get(svg_url).content
    png = svg2png(bytestring=svg_data)
    pil_image = image.open(BytesIO(png))
    pil_image.show()

if __name__ == "__main__":
    svgToPng("C:\\Users\\80521\\Desktop\\download.svg")
