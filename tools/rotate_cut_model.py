import numpy as np
import plyfile
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def inspect_ply_structure(ply_data):
    """PLY 파일의 구조를 검사합니다."""
    print(ply_data)
    print("Elements:", ply_data.elements)
    for element in ply_data.elements:
        print(f"Element name: {element.name}, Number of properties: {len(element.properties)}")


def load_ply(ply_file_path):
    """PLY 파일을 로드하고 데이터를 반환합니다."""
    with open(ply_file_path, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
    return plydata


def save_ply(ply_data, output_file_path):
    """PLY 데이터를 새로운 파일에 저장합니다."""
    with open(output_file_path, 'wb') as f:
        ply_data.write(f)


def load_transformation_matrix(matrix_file_path):
    """텍스트 파일에서 4x4 변환 행렬을 로드합니다."""
    return np.loadtxt(matrix_file_path)


def apply_transformation(matrix, vec):
    """변환 행렬을 벡터에 적용합니다."""
    vec_homogeneous = np.append(vec, 1)  # 4D 벡터로 변환하여 행렬 곱셈 가능하게 함
    transformed_vec = np.dot(matrix, vec_homogeneous)
    return transformed_vec[:3]  # x, y, z 성분만 반환


def transform_quaternion(matrix, quaternion):
    """회전 행렬을 quaternion에 적용합니다."""
    # 회전 행렬을 Rotation 객체로 변환
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)  # Rotation 객체 생성

    # 입력 quaternion이 [w, x, y, z] 형식이라고 가정하고, scipy에 맞게 [x, y, z, w]로 변환
    vertex_quat = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    vertex_rotation = R.from_quat(vertex_quat)

    # 회전 적용 (rotation 뒤에 vertex_rotation 곱셈은 rotation을 먼저 적용하는 것을 의미)
    transformed_rotation = rotation * vertex_rotation

    # 결과 quaternion을 [x, y, z, w] 형식으로 얻음
    result_quat = transformed_rotation.as_quat()

    # [w, x, y, z] 형식으로 변환
    result_quat = np.array([result_quat[3], result_quat[0], result_quat[1], result_quat[2]])

    # quaternion 정규화
    norm = np.linalg.norm(result_quat)
    if norm != 0:
        result_quat /= norm

    return result_quat


def create_filtered_ply(ply_data, filtered_vertices):
    """필터링된 vertex로 새로운 PlyData 객체를 생성합니다."""
    vertex_element = ply_data['vertex']

    # 원본 vertex 배열과 동일한 dtype으로 새로운 구조화된 배열 생성
    filtered_array = np.array(filtered_vertices, dtype=vertex_element.data.dtype)

    # 필터링된 vertex로 새로운 PlyElement 생성
    vertex_element_filtered = plyfile.PlyElement.describe(filtered_array, 'vertex')

    # 필터링된 vertex로 새로운 PlyData 객체 생성
    new_ply_data = plyfile.PlyData([vertex_element_filtered], text=ply_data.text)

    return new_ply_data


def transform_ply(input_ply_path, matrix_file_path, output_ply_path, max_distance=8):
    """입력된 PLY 파일을 변환하고 결과를 저장합니다."""
    # PLY 파일과 변환 행렬 로드
    ply_data = load_ply(input_ply_path)
    matrix = load_transformation_matrix(matrix_file_path)

    # vertex 가져오기
    vertices = ply_data['vertex']

    # 원점으로부터의 거리에 따라 vertex 필터링 (변환 전 좌표 사용)
    filtered_vertices = []
    for vertex in vertices:
        # 원점으로부터의 거리 계산 (변환 전)
        distance = np.sqrt(vertex['x'] ** 2 + vertex['y'] ** 2 + vertex['z'] ** 2)
        if distance <= max_distance:
            filtered_vertices.append(vertex)

    print(f"원래 vertex 수: {len(vertices)}, 필터링된 vertex 수: {len(filtered_vertices)}")

    # 변환되고 필터링된 vertex를 저장할 리스트 준비
    transformed_vertices = []

    # 법선과 quaternion 변환을 위한 회전 행렬 및 Rotation 객체 미리 계산
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)

    for vertex in tqdm(filtered_vertices):
        # x, y, z 변환
        transformed_xyz = apply_transformation(matrix, [vertex['x'], vertex['y'], vertex['z']])

        # 원본 데이터를 수정하지 않기 위해 vertex 복사
        transformed_vertex = vertex.copy()

        # 좌표 업데이트
        transformed_vertex['x'], transformed_vertex['y'], transformed_vertex['z'] = transformed_xyz

        # 법선 nx, ny, nz 변환 (회전 행렬 사용)
        normal = np.array([vertex['nx'], vertex['ny'], vertex['nz']])
        transformed_normal = rotation.apply(normal)

        # 법선 벡터 정규화
        norm = np.linalg.norm(transformed_normal)
        if norm != 0:
            transformed_normal /= norm
        transformed_vertex['nx'], transformed_vertex['ny'], transformed_vertex['nz'] = transformed_normal

        # 회전 quaternion rot_0, rot_1, rot_2, rot_3 변환
        # quaternion이 [w, x, y, z] 형식이라고 가정
        original_quaternion = np.array([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']])
        transformed_quaternion = transform_quaternion(matrix, original_quaternion)
        transformed_vertex['rot_0'], transformed_vertex['rot_1'], transformed_vertex['rot_2'], transformed_vertex['rot_3'] = transformed_quaternion

        # 변환된 vertex를 리스트에 추가
        transformed_vertices.append(transformed_vertex)

    # 변환되고 필터링된 vertex로 새로운 PLY 파일 생성
    new_ply_data = create_filtered_ply(ply_data, transformed_vertices)

    # 필터링된 PLY 데이터를 새로운 파일에 저장
    save_ply(new_ply_data, output_ply_path)


# 사용 예시
# input_ply_path = "./plys/1.ply"
# matrix_file_path = "./matrix/1.txt"
# output_ply_path = "./output/1.ply"

# transform_ply(input_ply_path, matrix_file_path, output_ply_path, max_distance=8)