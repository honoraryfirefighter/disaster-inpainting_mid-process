import sys
sys.path.append('C:/Users/admin/Desktop/seg inpainting/clipseg_repo')

import torch
from torchvision import transforms
from PIL import Image
from models.clipseg import CLIPDensePredT
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

# 재난 유형에 따른 영어 단어 매핑
disaster_type_to_english = {
    '지진': 'earthquake',
    '지반침하': 'land subsidence',
    '싱크홀': 'sinkhole',
    '토석류': 'mudslide',
    '홍수': 'flood',
    '폭풍해일': 'storm surge'
}

# segment_prompt를 설정하는 함수
def get_segment_prompt(disaster_type):
    ground_related = ['지진', '지반침하', '싱크홀', '토석류']
    water_related = ['홍수', '폭풍해일']
    if disaster_type in ground_related:
        return 'ground'
    elif disaster_type in water_related:
        return 'ocean'
    return 'unknown'

def apply_inpainting(image_path, disaster_type, alert_intensity):
    # CLIP 모델과 Stable Diffusion 인페인팅 파이프라인 설정
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load('C:/Users/admin/Desktop/seg inpainting/clipseg_weights/clipseg_weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)
    # Stable Diffusion 모델 컴포넌트 임포트 및 초기화
    from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
    model_dir = "stabilityai/stable-diffusion-2-inpainting"  # Hugging Face Hub의 모델 식별자
    scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir, scheduler=scheduler, revision="fp16", torch_dtype=torch.float32)
    pipe = pipe.to("cpu")


    # 이미지를 로드하고 전처리합니다.
    source_image = Image.open(image_path).convert('RGB')

    # 인페인팅할 부분의 마스크를 생성합니다.
    tensor_image = transforms.ToTensor()(source_image).unsqueeze(0)
    segment_prompt = get_segment_prompt(disaster_type)  # 재난 유형에 따른 segment_prompt 설정
    clipseg_prompt = segment_prompt  # ClipSeg 프롬프트로 segment_prompt를 사용합니다.
    with torch.no_grad():
        preds = model(tensor_image, [clipseg_prompt])[0]
    processed_mask = torch.special.ndtr(preds[0][0])
    expanded_mask = expand_mask_based_on_alert_intensity(processed_mask, alert_intensity)
    stable_diffusion_mask = transforms.ToPILImage()(expanded_mask)

    # 인페인팅 프롬프트 설정
    english_disaster_type = disaster_type_to_english.get(disaster_type, 'disaster')
    inpainting_prompt = f"{english_disaster_type} is occurred"  # 영어 단어를 사용

    # 이미지 인페인팅을 수행합니다.
    generator = torch.Generator(device="cpu").manual_seed(77)
    result_image = pipe(prompt=inpainting_prompt, guidance_scale=7.5, num_inference_steps=60, generator=generator, image=source_image, mask_image=stable_diffusion_mask).images[0]

    return result_image

# 마스크 확장 함수
def expand_mask_based_on_alert_intensity(processed_mask, alert_intensity):
    kernel_size = 20
    if alert_intensity == '주의보':
        kernel_size = 40
    elif alert_intensity == '경보':
        kernel_size = 60
    expanded_mask = torch.nn.functional.max_pool2d(processed_mask.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze()
    return (expanded_mask - expanded_mask.min()) / (expanded_mask.max() - expanded_mask.min())

# 함수 사용 예:
# result_image = apply_inpainting('path/to/image.jpg', '지진', '경보')

