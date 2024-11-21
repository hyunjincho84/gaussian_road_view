import struct


def read_ply_header(file_path):
    """헤더를 읽고 element vertex 수와 header 크기를 반환합니다."""
    with open(file_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii')
            header.append(line.strip())
            if line.startswith("end_header"):
                break

        vertex_count = None
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])

        header_size = f.tell()  # 헤더의 끝 위치 (데이터가 시작되는 위치)
        return vertex_count, header_size, header


def merge_ply(file1_path, file2_path, output_path):
    # 첫 번째 파일의 헤더 및 vertex 정보를 가져옵니다.
    vertex_count1, header_size1, header1 = read_ply_header(file1_path)

    # 두 번째 파일의 헤더 및 vertex 정보를 가져옵니다.
    vertex_count2, header_size2, header2 = read_ply_header(file2_path)

    # 합친 파일의 새로운 헤더 생성
    total_vertex_count = vertex_count1 + vertex_count2
    new_header = []
    for line in header1:
        if line.startswith("element vertex"):
            new_header.append(f"element vertex {total_vertex_count}")
        else:
            new_header.append(line)

    # 바이너리 데이터 크기 계산 (float는 4바이트)
    data_size_per_vertex = 4 * (3 + 3 + 3 + 46 + 1 + 3 + 4)  # 각 vertex의 속성 수

    with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2, open(output_path, 'wb') as fout:
        # 새 헤더를 씁니다.
        fout.write("\n".join(new_header).encode('ascii') + b'\n')

        # 첫 번째 파일의 데이터를 헤더 이후부터 읽고 씁니다.
        f1.seek(header_size1)
        fout.write(f1.read(vertex_count1 * data_size_per_vertex))

        # 두 번째 파일의 데이터를 헤더 이후부터 읽고 씁니다.
        f2.seek(header_size2)
        fout.write(f2.read(vertex_count2 * data_size_per_vertex))

    print(f"PLY 파일이 {output_path}에 성공적으로 병합되었습니다.")


# 사용 예시
# file1 = './two.ply'
# file2 = './output/1.ply'
# output_file = './three.ply'

# merge_ply(file1, file2, output_file)