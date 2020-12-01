import json
import math
from typing import List


class KinectJoint(object):
    def __init__(self) -> None:
        self.x_joint = 0.0
        self.y_joint = 0.0
        self.z_joint = 0.0

    def __init__(self, x_joint: float, y_joint: float, z_joint: float) -> None:
        self.x_joint = x_joint
        self.y_joint = y_joint
        self.z_joint = z_joint


class BodyData(object):
    def __init__(self) -> None:
        self.higher_x = -math.inf
        self.lower_x = math.inf
        self.higher_y = -math.inf
        self.lower_y = math.inf
        self.higher_z = -math.inf
        self.lower_z = math.inf

    def compute_higher_lower_values(self, x_joint: float, y_joint: float, z_joint: float, ) -> None:
        if x_joint > self.higher_x:
            self.higher_x = x_joint
        if x_joint < self.lower_x:
            self.lower_x = x_joint
        if y_joint > self.higher_y:
            self.higher_y = y_joint
        if y_joint < self.lower_y:
            self.lower_y = y_joint
        if z_joint > self.higher_z:
            self.higher_z = z_joint
        if z_joint < self.lower_z:
            self.lower_z = z_joint


class KinectBody(object):
    def __init__(self) -> None:
        self.body_id = ''
        self.joint_data = []

    def __init__(self, body_id: str, joint_data: List[KinectJoint]) -> None:
        self.body_id = body_id
        self.joint_data = joint_data

    def __del__(self) -> None:
        del self.joint_data


class KinectBlock(object):
    def __init__(self, n_bodies: int, n_joints: int, body_list: List[KinectBody]) -> None:
        self.n_bodies = n_bodies
        self.n_joints = n_joints
        self.body_list = body_list


class KinectData(object):
    kinect_blocks: List[KinectBlock]
    perturbation_percent = 0.05

    def __init__(self) -> None:
        self.n_frames = 0
        self.n_joints = 0
        self.n_bodies = 0
        self.kinect_blocks = []
        self.n_bodies = 0
        self.body_data = []

    def __del__(self) -> None:
        del self.kinect_blocks

    def check_n_bodies(self, n_bodies: int) -> None:
        if self.n_bodies < n_bodies:
            for _ in range(n_bodies - self.n_bodies):
                self.body_data.append(BodyData())
            self.n_bodies = len(self.body_data)

    def read_block_NTU(self, file) -> None:
        """Read NTU block of Kinect data."""
        n_bodies = int(file.readline())
        self.check_n_bodies(n_bodies)
        body_list = []
        for i_body in range(n_bodies):
            data = file.readline()
            split_str = data.split(' ')
            body_id = split_str[0]
            n_joints = int(file.readline())
            joint_data = []
            for i_joint in range(n_joints):
                str_split = file.readline().split(' ')
                x_joint = float(str_split[0])
                y_joint = float(str_split[1])
                z_joint = float(str_split[2])
                joint_data.append(KinectJoint(x_joint, y_joint, z_joint))
                self.body_data[i_body].compute_higher_lower_values(x_joint, y_joint, z_joint)
            kinect_body = KinectBody(body_id, joint_data)
            body_list.append(kinect_body)
        if n_bodies > 0:
            kb = KinectBlock(n_bodies, n_joints, body_list)
        else:
            n_joints = 25
            kb = KinectBlock(n_bodies, n_joints, body_list)
        self.kinect_blocks.append(kb)
        if kb.n_bodies > self.n_bodies:
            self.n_bodies = kb.n_bodies

    def read_data(self, skl_file: str) -> None:
        """Read the Kinect data from NTU skeleton file."""
        file = open(skl_file, 'r')
        n_frames = int(file.readline())
        for _ in range(n_frames):
            self.read_block_NTU(file)
        self.n_joints = self.kinect_blocks[0].n_joints  # NTU = 25
        self.n_frames = len(self.kinect_blocks)  # Get by blocks, because there are some frames without skeleton data
        file.close()


class OpenPoseData(object):
    kinect_blocks: List[KinectBlock]
    perturbation_percent = 0.05

    def __init__(self) -> None:
        self.n_frames = 0
        self.n_joints = 0
        self.n_bodies = 0
        self.kinect_blocks = []
        self.n_bodies = 0
        self.body_data = []

    def __del__(self) -> None:
        del self.kinect_blocks

    def check_n_bodies(self, n_bodies: int) -> None:
        if self.n_bodies < n_bodies:
            for _ in range(n_bodies - self.n_bodies):
                self.body_data.append(BodyData())
            self.n_bodies = len(self.body_data)

    def read_block_NTU(self, file) -> None:
        """Read NTU block of Kinect data."""
        n_bodies = int(file.readline())
        self.check_n_bodies(n_bodies)
        body_list = []
        for i_body in range(n_bodies):
            data = file.readline()
            split_str = data.split(' ')
            body_id = split_str[0]
            n_joints = int(file.readline())
            joint_data = []
            for i_joint in range(n_joints):
                str_split = file.readline().split(' ')
                x_joint = float(str_split[0])
                y_joint = float(str_split[1])
                z_joint = float(str_split[2])
                joint_data.append(KinectJoint(x_joint, y_joint, z_joint))
                self.body_data[i_body].compute_higher_lower_values(x_joint, y_joint, z_joint)
            kinect_body = KinectBody(body_id, joint_data)
            body_list.append(kinect_body)
        if n_bodies > 0:
            kb = KinectBlock(n_bodies, n_joints, body_list)
        else:
            n_joints = 25
            kb = KinectBlock(n_bodies, n_joints, body_list)
        self.kinect_blocks.append(kb)
        if kb.n_bodies > self.n_bodies:
            self.n_bodies = kb.n_bodies

    def read_block_OP(self, skeleton_data_frame) -> None:
        """Read frame of open pose data."""
        n_bodies = len(skeleton_data_frame["skeleton"])
        self.check_n_bodies(n_bodies)
        body_list = []
        for i_body in range(n_bodies):
            data = skeleton_data_frame["skeleton"][i_body]
            n_joints = int(len(data["pose"]) / 2)
            assert n_joints == 18

            joints_xys = skeleton_data_frame["skeleton"][i_body]["pose"]
            j_xs = [v for idx, v in enumerate(joints_xys) if idx % 2 == 0]
            j_ys = [-v for idx, v in enumerate(joints_xys) if idx % 2 == 1]

            joint_data = []
            for i_joint in range(n_joints):
                x_joint = j_xs[i_joint]
                y_joint = j_ys[i_joint]
                z_joint = 0.0
                joint_data.append(KinectJoint(x_joint, y_joint, z_joint))
                self.body_data[i_body].compute_higher_lower_values(x_joint, y_joint, z_joint)
            kinect_body = KinectBody(i_body, joint_data)
            body_list.append(kinect_body)
        if n_bodies > 0:
            kb = KinectBlock(n_bodies, n_joints, body_list)
        else:
            n_joints = 18
            kb = KinectBlock(n_bodies, n_joints, body_list)
        self.kinect_blocks.append(kb)
        if kb.n_bodies > self.n_bodies:
            self.n_bodies = kb.n_bodies

    def read_data(self, file: str) -> None:
        """Read the Kinect data from NTU skeleton file."""
        with open(file) as f:
            skeleton_info = json.load(f)
            n_frames = len(skeleton_info["data"])

            for i in range(n_frames):
                self.read_block_OP(skeleton_info["data"][i])

            self.n_joints = self.kinect_blocks[0].n_joints  # OP = 18

            # Get by blocks, because there are some frames without skeleton data
            self.n_frames = len(self.kinect_blocks)


def from_format_desc(input_format):
    in_types = {"nturgbd_csv":   KinectData,
                "openpose_json": OpenPoseData}

    return in_types[input_format]()


def skeleton_data(kd, i, scat, ax, ann_list):
    body_blocks = next((body for body in kd.kinect_blocks[i].body_list if body.body_id == 0), None)

    if body_blocks is None:
        return

    jdat = body_blocks.joint_data

    xs = [j.x_joint for j in jdat]
    ys = [j.y_joint for j in jdat]

    import numpy as np

    xys = np.stack([xs, ys])
    xys = xys.transpose()
    # zs = [j.z_joint for j in jdat]

    scat.set_offsets(xys)

    for i, a in enumerate(ann_list):
        try:
            a.set_position((xs[i] + 0.01, ys[i] - 0.01))
        except IndexError as err:
            print(err)

    ax.legend()
    ax.grid(True)

    return scat

def show_data(path, dtype="kinect"):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()

    # creating a subplot
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))

    ann_list = []

    scat = ax.scatter([], [], alpha=0.3, edgecolors='none')

    kd = None

    with open(path) as f:
        if dtype == "openpose":
            kd = OpenPoseData()
            kd.read_data(path)
        elif dtype == "kinect":
            kd = KinectData()
            kd.read_data(path)
        else:
            raise ValueError

    if kd is None:
        raise ValueError

    def init():
        scat.set_offsets([])

        if len(ann_list) == 0:
            n = [str(i) for i in range(kd.kinect_blocks[0].n_joints)]

            for txt in n:
                ann_list.append(ax.annotate(txt, (0, 0)))

        return scat,

    updates = lambda i: skeleton_data(kd, i, scat, ax, ann_list)

    ani = animation.FuncAnimation(fig, updates, interval=1000./10, frames=len(kd.kinect_blocks), init_func=init)
    plt.show()


def main():
    show_data("/home/david/datasets/kinetics/kinetics400-skeleton/kinetics_val/Vhf92EnnS7o.json", dtype="openpose")
    #show_data("/home/david/datasets/nturgbd/skeleton_csv/S001C001P001R001A009.skeleton", dtype="kinect")

    "/media/david/Daten/datasets/kinetics/kinetics-skeleton/kinetics_train/0074cdXclLU.json"


if __name__ == '__main__':
    main()
