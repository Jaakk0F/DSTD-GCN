import numpy as np


class Graph:

    def __init__(self, layout="h36m"):
        self.load_graph(layout)

    def load_graph(self, layout):
        if layout == "h36m":
            # this is 22 point version
            bone_pair = [
                (5, 4),
                (10, 9),
                (4, 3),
                (9, 8),
                (3, 2),
                (8, 7),
                (13, 12),
                (14, 12),
                (21, 19),
                (22, 19),
                (19, 18),
                (29, 27),
                (30, 27),
                (27, 26),
                (18, 17),
                (26, 25),
                (17, 13),
                (25, 13),
                (14, 13),
                (15, 14),
                # self-defined
                (2, 12),
                (7, 12),
            ]
            use_joint = {
                2: 0,
                3: 1,
                4: 2,
                5: 3,
                7: 4,
                8: 5,
                9: 6,
                10: 7,
                12: 8,
                13: 9,
                14: 10,
                15: 11,
                17: 12,
                18: 13,
                19: 14,
                21: 15,
                22: 16,
                25: 17,
                26: 18,
                27: 19,
                29: 20,
                30: 21,
            }
            # this is some manually defined connection, like two arms and arms and legs
            part_pair = {
                # mirror
                (17, 25),
                (18, 26),
                (19, 27),
                (21, 29),
                (22, 30),
                (2, 7),
                (3, 8),
                (4, 9),
                (5, 10),
                # arm & leg
                (18, 2),
                (26, 7),
                (18, 7),
                (26, 2),
                (19, 3),
                (27, 8),
                (19, 8),
                (27, 3),
            }
            self.num_joint = len(use_joint)
            self.bone_pair = []
            self.part_pair = []
            for bp in bone_pair:
                self.bone_pair.append([use_joint[bp[0]], use_joint[bp[1]]])
            for pp in part_pair:
                self.part_pair.append([use_joint[pp[0]], use_joint[pp[1]]])
        elif layout == "cmu":
            # this is 25 point version
            bone_pair = [
                # foot
                (6, 5),
                (5, 4),
                (4, 3),
                (10, 9),
                (11, 10),
                (12, 11),
                # body
                (15, 14),
                (17, 15),
                (18, 17),
                (19, 18),
                # arms
                (30, 15),
                (31, 30),
                (32, 31),
                (34, 32),
                (35, 34),
                (37, 32),
                (26, 25),
                (25, 23),
                (28, 23),
                (23, 22),
                (22, 21),
                (21, 15),
                # self-defined
                (9, 14),
                (3, 14)
            ]
            use_joint = {
                3: 0,
                4: 1,
                5: 2,
                6: 3,
                9: 4,
                10: 5,
                11: 6,
                12: 7,
                14: 8,
                15: 9,
                17: 10,
                18: 11,
                19: 12,
                21: 13,
                22: 14,
                23: 15,
                25: 16,
                26: 17,
                28: 18,
                30: 19,
                31: 20,
                32: 21,
                34: 22,
                35: 23,
                37: 24,
            }
            part_pair = {
                # mirror
                (30, 21),
                (31, 22),
                (32, 23),
                (37, 28),
                (34, 25),
                (35, 26),
                (9, 3),
                (10, 4),
                (11, 5),
                (12, 4),
                # arm refine
                (21, 23),
                (21, 25),
                (21, 26),
                (21, 28),
                (25, 28),
                (26, 28),
                (30, 32),
                (30, 34),
                (30, 35),
                (30, 37),
                (34, 37),
                (35, 37),
                (22, 30),
                (21, 31),
                (23, 31),
                (22, 32),
                # leg refine
                (3, 5),
                (3, 6),
                (4, 6),
                (9, 11),
                (9, 12),
                (10, 12),
                (4, 9),
                (3, 10),
                # leg & arm
                (31, 9),
                (22, 3),
                (32, 10),
                (23, 4),
                (31, 3),
                (23, 9),
                (22, 10),
                (31, 4),
                (32, 9),
                (32, 3),
                (23, 3),
                (23, 9),
            }
            self.num_joint = len(use_joint)
            self.bone_pair = []
            self.part_pair = []
            for bp in bone_pair:
                self.bone_pair.append([use_joint[bp[0]], use_joint[bp[1]]])
            for pp in part_pair:
                self.part_pair.append([use_joint[pp[0]], use_joint[pp[1]]])
        elif layout == "expi":
            # 18 joint per person
            I = np.array([0, 0, 0, 3, 4, 6, 3, 5, 7, 3, 10, 12, 14, 3, 11, 13, 15])
            J = np.array([1, 2, 3, 4, 6, 8, 5, 7, 9, 10, 12, 14, 16, 11, 13, 15, 17])
            self.num_joint = 2 * (max(np.max(I), np.max(J)) + 1)
            self.bone_pair = []
            self.part_pair = []
            n_iter = len(I)
            for i in range(n_iter):
                self.bone_pair.extend([[I[i], J[i]], [I[i] + self.num_joint // 2, J[i] + self.num_joint // 2]])
        elif layout == "3dpw":
            # this is 25 point version
            bone_pair = [
                # foot
                (1, 4),
                (4, 7),
                (7, 10),
                (2, 5),
                (5, 8),
                (8, 11),
                # body
                (1, 3),
                (2, 3),
                (3, 6),
                (6, 9),
                (9, 12),
                (9, 13),
                (9, 14),
                (12, 13),
                (12, 14),
                (12, 15),
                # arms
                (13, 16),
                (14, 17),
                (16, 18),
                (17, 19),
                (18, 20),
                (19, 21),
                (20, 22),
                (21, 23),
                # self-defined
                (9, 14),
                (3, 14),
            ]
            use_joint = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                9: 8,
                10: 9,
                11: 10,
                12: 11,
                13: 12,
                14: 13,
                15: 14,
                16: 15,
                17: 16,
                18: 17,
                19: 18,
                20: 19,
                21: 20,
                22: 21,
                23: 22,
            }
            part_pair = {
                # mirror
                (1, 2),
                (4, 5),
                (7, 8),
                (10, 11),
                (13, 14),
                (16, 17),
                (18, 19),
                (20, 21),
                (22, 23),
                # leg & arm
                (18, 4),
                (19, 5),
                (20, 7),
                (21, 8),
                (22, 10),
                (23, 11),
            }
            self.num_joint = len(use_joint)
            self.bone_pair = []
            self.part_pair = []
            for bp in bone_pair:
                self.bone_pair.append([use_joint[bp[0]], use_joint[bp[1]]])
            for pp in part_pair:
                self.part_pair.append([use_joint[pp[0]], use_joint[pp[1]]])
        else:
            raise NotImplementedError()

    def get_adjacency(self):
        adj = np.eye(self.num_joint)
        for bp in self.bone_pair:
            # connection of different joint
            adj[bp[0], bp[1]] = 1
            adj[bp[1], bp[0]] = 1
        for pp in self.part_pair:
            adj[pp[0], pp[1]] = 1
            adj[pp[1], pp[0]] = 1
        return adj

    def get_adjacency_type(self, type="self"):
        if type == "self":
            adj = np.eye(self.num_joint)
        elif type == "connect":
            adj = np.eye(self.num_joint)
            # adj = np.zeros((self.num_joint, self.num_joint))
            for bp in self.bone_pair:
                # connection of different joint
                adj[bp[0], bp[1]] = 1
                adj[bp[1], bp[0]] = 1
        elif type == "part":
            adj = np.zeros((self.num_joint, self.num_joint))
            for pp in self.part_pair:
                adj[pp[0], pp[1]] = 1
                adj[pp[1], pp[0]] = 1
        elif type == "all":
            # self
            adj = np.eye(self.num_joint)
            # bone
            for bp in self.bone_pair:
                # connection of different joint
                adj[bp[0], bp[1]] = 1
                adj[bp[1], bp[0]] = 1
            # semantic
            for pp in self.part_pair:
                adj[pp[0], pp[1]] = 1
                adj[pp[1], pp[0]] = 1
        else:
            raise ValueError(f"Invalid graph type {type}")
        return adj

    def get_all_adjacency(self):
        adj = []
        # adj.append(self.get_adjacency_type("self"))
        adj.append(self.get_adjacency_type("connect"))
        adj.append(self.get_adjacency_type("part"))
        # adj.append(self.get_adjacency_type("all"))
        adj = np.stack(adj, axis=0)
        return adj


class GraphJBC:
    # Joint Bone Cross
    def __init__(self, layout="h36m") -> None:
        self.load_graph(layout)

    def load_graph(self, layout):
        if layout == "h36m":
            # this is 22 point version
            bone_pair = [
                (5, 4),
                (10, 9),
                (4, 3),
                (9, 8),
                (3, 2),
                (8, 7),
                (13, 12),
                (14, 12),
                (21, 19),
                (22, 19),
                (19, 18),
                (29, 27),
                (30, 27),
                (27, 26),
                (18, 17),
                (26, 25),
                (17, 13),
                (25, 13),
                (14, 13),
                (15, 14),
            ]
            use_joint = {
                2: 0,
                3: 1,
                4: 2,
                5: 3,
                7: 4,
                8: 5,
                9: 6,
                10: 7,
                12: 8,
                13: 9,
                14: 10,
                15: 11,
                17: 12,
                18: 13,
                19: 14,
                21: 15,
                22: 16,
                25: 17,
                26: 18,
                27: 19,
                29: 20,
                30: 21,
            }
            self.num_joint = len(use_joint)
            self.num_bone = len(bone_pair)
            self.bone_pair = []
            for bp in bone_pair:
                self.bone_pair.append([use_joint[bp[0]], use_joint[bp[1]]])
        elif layout == "cmu":
            pass
        else:
            raise NotImplementedError()

    def get_joint_adjacency(self):
        joint_adj = np.eye(self.num_joint)
        for bp in self.bone_pair:
            # connection of different joint
            joint_adj[bp[0], bp[1]] = 1
            joint_adj[bp[1], bp[0]] = 1
        # return normalize_digraph(joint_adj)
        return joint_adj

    def get_bone_adjacency(self):
        # get bone adjacency matrix from bone
        num_bone = len(self.bone_pair)
        bone_adj = np.eye(num_bone)
        for ib in range(num_bone):
            for jb in range(ib, num_bone):
                if (len(set(self.bone_pair[ib]) & set(self.bone_pair[jb])) > 0):
                    bone_adj[ib, jb] = 1
        # return normalize_digraph(bone_adj)
        return bone_adj

    def get_cross_adjacency(self):
        # from bone to joint
        num_joint = self.num_joint
        num_bone = self.num_bone
        cross_adj = np.zeros((num_bone, num_joint))
        for i in range(num_bone):
            cross_adj[i, self.bone_pair[i][0]] = 1
            cross_adj[i, self.bone_pair[i][1]] = 1
        return cross_adj


class GraphFlatten:
    # Joint Bone Cross
    def __init__(self, layout="h36m") -> None:
        self.load_graph(layout)

    def load_graph(self, layout):
        if layout == "h36m":
            # this is 22 point version
            bone_pair = [
                (5, 4),
                (10, 9),
                (4, 3),
                (9, 8),
                (3, 2),
                (8, 7),
                (13, 12),
                (14, 12),
                (21, 19),
                (22, 19),
                (19, 18),
                (29, 27),
                (30, 27),
                (27, 26),
                (18, 17),
                (26, 25),
                (17, 13),
                (25, 13),
                (14, 13),
                (15, 14),
            ]
            use_joint = {
                2: 0,
                3: 1,
                4: 2,
                5: 3,
                7: 4,
                8: 5,
                9: 6,
                10: 7,
                12: 8,
                13: 9,
                14: 10,
                15: 11,
                17: 12,
                18: 13,
                19: 14,
                21: 15,
                22: 16,
                25: 17,
                26: 18,
                27: 19,
                29: 20,
                30: 21,
            }
            self.num_joint = len(use_joint)
            self.num_bone = len(bone_pair)
            self.bone_pair = []
            for bp in bone_pair:
                self.bone_pair.append([use_joint[bp[0]], use_joint[bp[1]]])
            self.bone_pair = np.array(self.bone_pair)
        elif layout == "cmu":
            pass
        else:
            raise NotImplementedError()

    def load_joint_graph(self):
        adj = np.zeros((self.num_joint * 3, self.num_joint * 3))
        adj[self.bone_pair[:, 0] * 3, self.bone_pair[:, 1] * 3] = 1
        adj[self.bone_pair[:, 0] * 3 + 1, self.bone_pair[:, 1] * 3 + 1] = 1
        adj[self.bone_pair[:, 0] * 3 + 2, self.bone_pair[:, 1] * 3 + 2] = 1
        adj[self.bone_pair[:, 1] * 3, self.bone_pair[:, 0] * 3] = 1
        adj[self.bone_pair[:, 1] * 3 + 1, self.bone_pair[:, 0] * 3 + 1] = 1
        adj[self.bone_pair[:, 1] * 3 + 2, self.bone_pair[:, 0] * 3 + 2] = 1
        return adj

    def load_coordinate_graph(self):
        adj = np.zeros((self.num_joint * 3, self.num_joint * 3))
        joint_arange = np.arange(self.num_joint)
        adj[joint_arange * 3, joint_arange * 3 + 1] = 1
        adj[joint_arange * 3, joint_arange * 3 + 2] = 1
        adj[joint_arange * 3 + 1, joint_arange * 3 + 2] = 1
        adj[joint_arange * 3 + 1, joint_arange * 3] = 1
        adj[joint_arange * 3 + 2, joint_arange * 3] = 1
        adj[joint_arange * 3 + 2, joint_arange * 3 + 1] = 1
        return adj

    def load_connection_graph(self):
        adj = np.zeros((self.num_joint * 3, self.num_joint * 3))
        joint_arange = np.arange(self.num_joint)
        adj[::3, joint_arange * 3] = 1
        adj[1::3, joint_arange * 3 + 1] = 1
        adj[2::3, joint_arange * 3 + 2] = 1
        adj[joint_arange * 3, ::3] = 1
        adj[joint_arange * 3 + 1, 1::3] = 1
        adj[joint_arange * 3 + 2, 2::3] = 1
        adj -= np.eye(self.num_joint * 3)
        return adj
