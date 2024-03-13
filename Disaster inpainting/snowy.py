from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random
import sys

def create_snow_effect(image_path):
    # 기본 이미지 불러오기
    base_image = Image.open(image_path)
    base_image = base_image.convert("RGBA")
    
    # 새 눈송이 레이어 생성
    snow_layer = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    
    # 눈송이 생성 및 랜덤 블러 적용
    for _ in range(4000):  # 눈송이의 개수
        temp_layer = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(temp_layer)
    
        x, y = random.randint(0, snow_layer.width), random.randint(0, snow_layer.height)
        radius = random.randint(1, 7)  # 눈송이의 크기
        blur_radius = random.uniform(1.0, 4.0)  # 랜덤 블러 반경
    
        draw.ellipse((x, y, x + radius, y + radius), fill=(255, 255, 255, 150))  # 반투명 흰색 눈송이
        temp_layer = temp_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
        snow_layer = Image.alpha_composite(snow_layer, temp_layer)
    
    # 눈송이 레이어와 기본 이미지 합성
    final_image = Image.alpha_composite(base_image, snow_layer)
    return final_image

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    final_image = create_snow_effect(image_path)
    final_image.show()

# pip install Pillow
# python snowy.py C:/Users/admin/Desktop/test/beautiful_home.jpg
