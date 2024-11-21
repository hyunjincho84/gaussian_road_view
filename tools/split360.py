import subprocess
import os
import shutil

def run_colmap_command_with_logging(cmd):
    # 프로세스 시작
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True) as proc:
        # 실시간으로 로그 출력
        for line in iter(proc.stdout.readline, ''):
            print(line, end='')
        # 프로세스 종료 대기
        proc.wait()
        return proc.returncode
    
def split_images_cmd(cmd, input, output):
    command = [cmd, '--equirectangularNbSplits','8', '-i', input, '-o', output, '--outSfMData', './' ]
    if run_colmap_command_with_logging(command) != 0:
        print("Error in feature extraction")
        
        
def data_construct(input, output):
    for root, dirs, files in os.walk(input):
        for file in files:
            if file.endswith('.jpg'):
                folder_number = os.path.basename(root)
            
                new_file_name = f"{os.path.splitext(file)[0]}_{folder_number}.jpg"
            
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(output, new_file_name)
            
                shutil.move(src_file_path, dst_file_path)
    
def split360_images(cmd, input, output, final_output):
    split_images_cmd(cmd,input, output)
    data_construct('./rig',final_output)
    