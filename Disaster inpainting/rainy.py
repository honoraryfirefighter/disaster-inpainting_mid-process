from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import cv2
import sys

def create_rain_effect(image_path):
    # 기본 이미지 불러오기 및 크기, 모드 조절
    base_image = Image.open(image_path)
    base_image = base_image.resize((512, 512))
    if base_image.mode != 'RGB':
        base_image = base_image.convert('RGB')

    # 검은색 배경 위에 백색 노이즈 생성
    noise = np.random.normal(loc=0, scale=1, size=(512, 512))
    noise_scaled = ((noise - noise.min()) / (noise.max() - noise.min()) * 255).astype('uint8')
    noise_image = Image.fromarray(noise_scaled)
    noise_image = noise_image.convert('L')  # L 모드로 변환

    # 모션 블러를 적용하기 위한 커널 생성
    kernel_size = 15
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size

    # 커널 회전을 위한 각도 설정
    angle = 45  # 원하는 각도로 설정

    # 커널 중심을 찾고 회전 매트릭스 생성
    center = (kernel_size // 2, kernel_size // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 회전된 커널 생성
    rotated_kernel_motion_blur = cv2.warpAffine(kernel_motion_blur, rot_mat, (kernel_size, kernel_size))

    # 모션 블러 적용
    noise_array = np.array(noise_image)  # 이미지를 numpy 배열로 변환
    rain_effect = cv2.filter2D(noise_array, -1, rotated_kernel_motion_blur)

    # numpy 배열을 PIL 이미지로 변환
    rain_effect_image = Image.fromarray(rain_effect)

    # 대비 및 밝기 조정
    contraster = ImageEnhance.Contrast(rain_effect_image)
    rain_effect_image = contraster.enhance(5.0)  # 대비를 높임
    enhancer = ImageEnhance.Brightness(rain_effect_image)
    enhanced_noise_image = enhancer.enhance(0.6)

    # enhanced_noise_image가 그레이스케일이므로 RGB 모드로 변환
    if enhanced_noise_image.mode == 'L':
        enhanced_noise_image = enhanced_noise_image.convert('RGB')

    # 스크린 블렌드 모드로 최종 이미지 합성
    final_image = screen_blend(enhanced_noise_image, base_image)
    return final_image

def screen_blend(top, bottom):
    """스크린 블렌드 모드를 구현하는 함수"""
    return ImageChops.invert(ImageChops.multiply(ImageChops.invert(top), ImageChops.invert(bottom)))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rainy.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    final_image = create_rain_effect(image_path)
    final_image.show(title="Final Image with Screen Blend")

# pip install Pillow numpy opencv-python
# python rainy.py C:/Users/admin/Desktop/test/house.jpg
