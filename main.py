import sys
from tools.vid_to_img import *
from tools.colmap_cmd import *
import os
from tools.get_merging_matrix import *
from tools.split360 import *
from tools.train import *
import numpy as np

from tools.merge import merge_ply  # 예시로 병합 함수가 tools 폴더에 있다고 가정
from tools.rotate_cut_model import transform_ply




def count_subdirectories(path):
    """주어진 경로에 있는 하위 디렉토리 수를 반환합니다."""
    # 주어진 경로에 있는 항목들의 리스트를 가져옵니다.
    entries = os.listdir(path)
    
    # entries 중에서 디렉토리인 것들만 필터링합니다.
    subdirectory_count = sum(os.path.isdir(os.path.join(path, entry)) for entry in entries)
    
    return subdirectory_count

def main():
    #colmap에서 만들어지는 database pwd
    database_path = 'database.db'

    #8방위로 찢은 이미지 dir
    image_path = './8_dir_frames'

    #colmap 과정 저장 dir
    output_path = './colmap_output'

    output_ply = './ply_files'

    image_list_path = './image_list'

    save_360_frames_path = './360frames'
    
    path_to_meshroom = '/home/mrlab/Meshroom-2023.3.0/aliceVision/bin/./aliceVision_split360Images'
    
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(save_360_frames_path, exist_ok=True)
    os.makedirs(image_list_path, exist_ok=True)
    os.makedirs('./matrix', exist_ok=True)
    os.makedirs('./cut_rotate', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    os.makedirs(output_ply, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    os.environ['LD_LIBRARY_PATH'] = '/home/mrlab/Meshroom-2023.3.0/aliceVision/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['ALICEVISION_ROOT'] = '/home/mrlab/Meshroom-2023.3.0/aliceVision'
    
    image_num = vid_to_img(sys.argv[1],save_360_frames_path)

    # # 360split하는거
    split360_images(path_to_meshroom, save_360_frames_path, './', image_path)
    print("!!!done splitting 360 images!!!")
    
    
    #data split을 해줌
    ##############327 image_num으로 무조건 바꿔야함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    total_image_num = image_num * 8
    submodel_image_num = 280
    merge_num = 80
    tmp = submodel_image_num
    submodel_num = 1
    while(tmp < total_image_num):
        tmp += (submodel_image_num - merge_num)
        submodel_num += 1
    print(submodel_num)
    # 왜 submodel_num -1을 넣냐면 마지막 꼬랑지가 scale이 안맞아서 merge가 이쁘게 안됨 그래서 걍 버림
    create_overlapping_image_lists(image_path,submodel_num - 1,submodel_image_num,merge_num)
    print("done creating submodel text files")
    
    
    
    def extract_number(filename):
        # 파일 이름에서 숫자 부분만 추출
        number = ''.join([char for char in filename if char.isdigit()])
        return int(number) if number else 0

    
    i = 1
    for (root2, dirs2, files2) in os.walk(image_list_path):
        files2 = sorted(files2, key=extract_number)
        for file2 in files2:
            mkdir(f"{output_path}/{i}")
            print(f"makeing progress with {i}th submodel...")
            feature_extractor(f"{output_path}/{i}/{database_path}", image_path, os.path.join(root2, file2))
            print(f"makeing progress with {i}th submodel...")
            exhaustive_matcher(f"{output_path}/{i}/{database_path}")
            mapper(f"{output_path}/{i}/{database_path}", image_path,f"{output_path}/{i}",os.path.join(root2, file2))
            print("***************")
            print("***************")
            print("***************")
            print("***************")
            print("***************")
            print(f"model {i} done")
            print("***************")
            print("***************")
            print("***************")
            print("***************")
            print("***************")
            a = count_subdirectories(f"{output_path}/{i}")
            if a >= 2:
                image_undistorter(image_path, f"{output_path}/{i}/{a-1}", f"{output_path}/{i}/undistorted",os.path.join(root2, file2))
            else:
                image_undistorter(image_path, f"{output_path}/{i}/0", f"{output_path}/{i}/undistorted", os.path.join(root2, file2))
            convert_colmap_bin_to_txt(f"{output_path}/{i}/undistorted/sparse/0", f"{output_path}/{i}/undistorted")
            export_model_to_ply(f"{output_path}/{i}/undistorted/sparse/0", f"{output_ply}/{i}.ply")
            
            i+=1

    # 위에까지 colmap 돌리고 .ply파일 저장함
    submodel_num = i
    matrixs = []
    
    matrixs.append(np.array([[1.,0.,0.,0.],
                             [0.,1.,0.,0.],
                             [0.,0.,1.,0.],
                             [0.,0.,0.,1.]]))
    
    for j in range(submodel_num - 2):
        matrixs.append(np.array(get_merge_matrix(f"{output_path}/{j+1}/undistorted/images.txt",f"{output_path}/{j+2}/undistorted/images.txt")))

   
    for k in range(len(matrixs) - 1):
        matrixs[k+1] = np.dot(matrixs[k],matrixs[k+1])
    #이제 matrix들은 준비 된거임.
    
    for i in range(len(matrixs)):
        path = f'./matrix/{i+1}.txt'
        with open(path, 'w') as file:
            for row in matrixs[i]:
                for num in row:
                    file.write(str(num) + '\t')
                file.write('\n')

    for f in range(len(matrixs)):
        print(matrixs[f])
    print("-------------------------")
    
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.8/lib64:/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    try:
        import torch
    except OSError as e:
        print("OSError:", e)

    train(submodel_num - 1)
    # 여기까지 gaussian 모델 만들었고 밑에서 부터 하나의 ply로 합치는 과정임.
    gaussian_path = './output'
    cut_rotate = './cut_rotate'
    result_prefix = './result'  # 중간 결과 파일의 prefix
    final_output = './result.ply'  # 최종 결과 파일 이름
        
    for number_dir in sorted(os.listdir(gaussian_path), key=lambda x: int(x)):
        number_path = os.path.join(gaussian_path, number_dir)
        point_cloud_dir = os.path.join(number_path, "point_cloud")
        
        if os.path.isdir(point_cloud_dir):
            ply = os.path.join(point_cloud_dir, "iteration_30000", "point_cloud.ply")
            
            if os.path.exists(ply):
                print(f"Using: {ply}")
                
            else:
                ply = os.path.join(point_cloud_dir, "iteration_7000", "point_cloud.ply")
                if os.path.exists(ply):
                    print(f"Using: {ply}")
                else:
                    print(f"No suitable point_cloud.ply found in {number_dir}")
                    continue 
        else:
            continue

        matrix_path = f'./matrix/{number_dir}.txt'

        transform_ply(ply, matrix_path, f'{cut_rotate}/{number_dir}.ply')

    files = [f for f in sorted(os.listdir(cut_rotate)) if f.endswith('.ply')]

    if files:
        # 첫 두 파일 병합 후 임시 결과 저장
        temp_result = f"{result_prefix}_2.ply"
        merge_ply(os.path.join(cut_rotate, files[0]), os.path.join(cut_rotate, files[1]), temp_result)

        # 이후의 파일들을 순차적으로 병합하면서 중간 파일 저장
        for i in range(2, len(files)):
            new_temp_result = f"{result_prefix}_{i+1}.ply"  # 새로운 중간 결과 파일 이름
            merge_ply(temp_result, os.path.join(cut_rotate, files[i]), new_temp_result)
            temp_result = new_temp_result  # 다음 단계에서 사용할 결과 파일을 갱신

        # 최종 결과를 result.ply로 이름 변경
        os.rename(temp_result, final_output)
        print(f"최종 병합 결과가 {final_output}에 저장되었습니다.")

if __name__ == "__main__":
    main()