import numpy as np


class Time:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.input_length = 10
        self.output_length = seq_length - self.input_length

    def get_adjacency(self):
        adj = np.eye(self.seq_length)
        adj[:-1, 1:] = np.eye(self.seq_length - 1)
        adj[1:, :-1] = np.eye(self.seq_length - 1)
        return adj

    def get_adjacency_type(self, type="self"):
        if type == "self":
            adj = np.eye(self.seq_length)
        elif type == "neighboor":
            adj = np.eye(self.seq_length)
            adj[:-1, 1:] = np.eye(self.seq_length - 1)
            adj[1:, :-1] = np.eye(self.seq_length - 1)
        elif type == "inout":
            adj = np.zeros((self.seq_length, self.seq_length))
            adj[:self.input_length, self.input_length:] = 1
            adj[self.input_length:, :self.input_length] = 1
        elif type == "all":
            adj = np.eye(self.seq_length)
            adj[:-1, 1:] = np.eye(self.seq_length - 1)
            adj[1:, :-1] = np.eye(self.seq_length - 1)
            adj[:self.input_length, self.input_length:] = 1
            adj[self.input_length:, :self.input_length] = 1
        else:
            raise ValueError(f"Invalid graph type {type}")
        return adj

    def get_all_adjacency(self):
        adj = []
        adj.append(self.get_adjacency_type("neighboor"))
        adj = np.stack(adj, axis=0)
        return adj
