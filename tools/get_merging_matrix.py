import numpy as np

merge_num = 10

# mode = 0 for starting ply & mode = 1 for last ply
def get_quaternion_position_pairs(file_path, mode = 0):
    quaternion_position_pairs = []

    with open(file_path, 'r') as file:
        lines_with_ = []
        for line in file:
            if '.jpg' in line:
                lines_with_.append(line)

    if mode == 0:
        sorted_lines = sorted(lines_with_, key=lambda x: int(x.split()[-1].split('_')[0]))[-8*merge_num:]
    elif mode == 1:
        sorted_lines = sorted(lines_with_, key=lambda x: int(x.split()[-1].split('_')[0]))[:8*merge_num]
    # i = 0
    # for lines in sorted_lines:
    #     print(lines)
    #     i+=1
    
    # print(f"********************{i}***********************")
    # print("********************************************")
    # print("********************************************")
    for line in sorted_lines:
        numbers = line.split()
        tmp = ([float(numbers[1]), float(numbers[2]), float(numbers[3]), float(numbers[4])],
               [float(numbers[5]), float(numbers[6]), float(numbers[7])])
        quaternion_position_pairs.append(tmp)


    return quaternion_position_pairs



def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def get_camera_pos(quaternion_position_pairs):
    a = 0
    global merge_num
    camera_position = np.zeros((merge_num,4))

    for (qvec, tvec) in quaternion_position_pairs:
        R = qvec2rotmat(qvec)
        t = np.array(tvec)
        t = -R.T @ t

        T = np.column_stack((R, t))
        T = np.vstack((T, (0, 0, 0, 1)))

        camera_center = np.array([0, 0, 0, 1])
        camera_center_transformed = T @ camera_center

        camera_position[int(a / 8)] = camera_position[int(a / 8)] + camera_center_transformed

        a += 1

    return camera_position


def initialize_with_camera_position(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB

    U, _, Vt = np.linalg.svd(H)

    R_mat = Vt.T @ U.T

    if np.linalg.det(R_mat) < 0:
        Vt[2, :] *= -1
        R_mat = Vt.T @ U.T

    t = centroid_B.T - R_mat @ centroid_A.T

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t

    return T


def get_merge_matrix(imagestxt1, imagestxt2):
    quaternion_position_pairs1 = get_quaternion_position_pairs(imagestxt1,mode = 0)
    quaternion_position_pairs2 = get_quaternion_position_pairs(imagestxt2,mode = 1)


    camera_position1 = np.array(get_camera_pos(quaternion_position_pairs1) / 8)
    camera_position2 = np.array(get_camera_pos(quaternion_position_pairs2) / 8)

    print("**********************************")
    print(imagestxt1)
    print(camera_position1)
    print(imagestxt2)
    print(camera_position2)
    print("**********************************")
    
    mat1 = initialize_with_camera_position(camera_position2[:, :-1],camera_position1[:, :-1])
    #ply2에 mat1 dot해줘야함
    return mat1



