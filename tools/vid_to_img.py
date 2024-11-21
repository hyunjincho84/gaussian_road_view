import cv2
import os

def vid_to_img(vid_path, save_path):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("동영상 파일을 열 수 없습니다.")
        return -1

    i = 0
    frame_count = 0
    j = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i % 30 == 0:
            # 이미지에서 하단 20%를 잘라내기
            height, width = frame.shape[:2]

            filename = f"{save_path}/{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            frame_count += 1
            j += 1
            # if j > 300:
            #     print(f"총 {frame_count}개의 프레임을 저장했습니다.")
            #     break

        i += 1

    print(f"총 {frame_count}개의 프레임을 저장했습니다.")
    return frame_count

def create_overlapping_image_lists(image_dir, num_lists, images_per_list, overlap):
    # 이미지 파일들을 찾아서 리스트로 만듭니다. (png, jpg, jpeg 확장자를 가진 파일만)
    image_files = [img for img in os.listdir(image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    # 파일명에서 첫 번째 언더스코어('_') 앞의 숫자를 기준으로 정렬하는 함수
    def sort_key(filename):
        # 파일명에서 숫자 부분만 추출하여 정렬 기준으로 사용
        number_part = filename.split('_')[0]
        return int(number_part)

    # 이미지 파일들을 정렬합니다.
    image_files = sorted(image_files, key=sort_key)

    if len(image_files) < images_per_list:
        raise ValueError("Not enough images to create the lists")

    # 이미지 리스트 파일들의 경로를 저장할 디렉터리가 없으면 생성합니다.
    os.makedirs('./image_list', exist_ok=True)
    
    # 이미지 리스트 생성
    for i in range(num_lists):
        # 리스트에 포함할 첫 이미지와 마지막 이미지의 인덱스 계산
        start_idx = i * (images_per_list - overlap)
        end_idx = start_idx + images_per_list
        image_list = image_files[start_idx:end_idx]

        # 이미지 리스트 파일 생성
        list_path = f'./image_list/image_list_{i+1}.txt'
        with open(list_path, 'w') as file:
            for image_name in image_list:
                file.write(image_name + '\n')
        print(f"Created {list_path} with {len(image_list)} images.")

