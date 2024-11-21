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

def activate_conda():
    cmd = ['conda', 'activate', 'gaussian_splatting']
    if run_colmap_command_with_logging(cmd) != 0:
        print("Error in feature extraction")
    
def construct_gaussian(input):
    cmd = ['python', '/home/mrlab/gaussian-splatting/train.py', '-s', input]#, '--output_name', output] 
    print("dsfafsafasf")
    if run_colmap_command_with_logging(cmd) != 0:
        print("Error in feature extraction")