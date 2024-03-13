import sys
from chatgpt import parse_disaster_alert
from rainy import create_rain_effect
from snowy import create_snow_effect
from background_inpainting import background_inpainting
from inpainting import apply_inpainting

def main(alert_text, image_path):
    # 재난 경보 텍스트를 분석합니다.
    parsed_alert = parse_disaster_alert(alert_text)
    disaster_type = parsed_alert['재난 종류']
    alert_intensity = parsed_alert.get('재난 강도', None)

    # 전역 규모 재난 또는 지표면 재난인지 결정합니다.
    global_disasters = ['태풍', '호우', '대설', '우박']
    surface_disasters = ['지진', '지반침하', '싱크홀', '토석류', '홍수', '폭풍해일']

    final_image = None

    if disaster_type in global_disasters:
        # 전역 규모 재난에 대한 배경 전처리를 적용합니다.
        print("전역 규모 재난에 대한 배경 전처리를 적용 중...")
        final_image = background_inpainting(image_path, disaster_type)
        
        # 필요한 경우 비 또는 눈 효과를 적용합니다.
        if disaster_type in ['태풍', '호우']:
            print("비 효과를 적용 중...")
            final_image = create_rain_effect(final_image)
        elif disaster_type in ['대설', '우박']:
            print("눈 효과를 적용 중...")
            final_image = create_snow_effect(final_image)

    elif disaster_type in surface_disasters:
        # 지표면 재난에 대한 인페인팅을 적용합니다.
        print(f"{disaster_type}에 대한 인페인팅을 적용 중...")
        final_image = apply_inpainting(image_path, disaster_type, alert_intensity)

    # 최종 이미지를 표시하거나 저장합니다.
    if final_image is not None:
        final_image.show()  # 최종 이미지를 표시합니다.
        # final_image.save('저장할_경로')  # 최종 이미지를 저장합니다.
    else:
        print("적용된 재난 효과가 없습니다.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py '<alert_text>' <image_path>")
        sys.exit(1)
    alert_text = sys.argv[1]
    image_path = sys.argv[2]
    main(alert_text, image_path)

