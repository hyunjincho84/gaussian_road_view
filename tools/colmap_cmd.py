import subprocess

def run_colmap_command_with_logging(cmd):
    # 프로세스 시작
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True) as proc:
        # 실시간으로 로그 출력
        for line in iter(proc.stdout.readline, ''):
            print(line, end='')
        # 프로세스 종료 대기
        proc.wait()
        return proc.returncode

def feature_extractor(database_path, image_path,image_list_path):
    cmd = ['colmap', 'feature_extractor',
           '--database_path', database_path,
           '--image_path', image_path,
            '--image_list_path', image_list_path]
    if run_colmap_command_with_logging(cmd) != 0:
        print("Error in feature extraction")

def exhaustive_matcher(database_path):
    cmd = ['colmap', 'exhaustive_matcher',
           '--database_path', database_path]
    if run_colmap_command_with_logging(cmd) != 0:
        print("Error in exhaustive matching")

def mkdir(file_name):
    cmd = ['mkdir', file_name]
    run_colmap_command_with_logging(cmd)

def mapper(database_path, image_path, output_path, image_list_path):
    cmd = ['colmap', 'mapper',
           '--database_path', database_path,
           '--image_path', image_path,
           '--output_path', output_path,
           '--image_list_path', image_list_path]
    if run_colmap_command_with_logging(cmd) != 0:
        print("Error in mapping")

def export_model_to_ply(input_path, output_path):
    cmd = ['colmap', 'model_converter',
           '--input_path', input_path,
           '--output_path', output_path,
           '--output_type', 'PLY']
    if run_colmap_command_with_logging(cmd) != 0:
        print("Error in exporting model to PLY")

def image_undistorter(image_path, input_path, output_path,image_list_path):
    cmd = ['mkdir', output_path]
    run_colmap_command_with_logging(cmd)
    cmd = ['colmap', 'image_undistorter',
     '--image_path', image_path,
     '--input_path', input_path,
     '--output_path', output_path,
      '--image_list_path', image_list_path]


    if run_colmap_command_with_logging(cmd) != 0:
        print("Error in exporting undistorted")
    
    cmd = ['mkdir', f"{output_path}/sparse/0"]
    run_colmap_command_with_logging(cmd)
    cmd = ['mv', f"{output_path}/sparse/cameras.bin", f"{output_path}/sparse/0"]
    run_colmap_command_with_logging(cmd)
    cmd = ['mv', f"{output_path}/sparse/images.bin", f"{output_path}/sparse/0"]
    run_colmap_command_with_logging(cmd)
    cmd = ['mv', f"{output_path}/sparse/points3D.bin", f"{output_path}/sparse/0"]
    run_colmap_command_with_logging(cmd)


def convert_colmap_bin_to_txt(input_folder, output_folder):
    cmd = ['colmap', 'model_converter',
           '--input_path', input_folder,
           '--output_path', output_folder,
           '--output_type', 'TXT']
    if run_colmap_command_with_logging(cmd) != 0:
        print("Error in exporting to TXT")

