from __future__ import annotations

from typing import List

import numpy as np


class PoseGraphOptimization:
    """Thin wrapper around g2o pose-graph optimization (optional dependency).

    This module is kept minimal for open-sourcing. The surrounding SLAM system integration
    (frontend, loop edges selection, outlier rejection, etc.) is not included.
    """

    def __init__(self) -> None:
        try:
            import g2o  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "PoseGraphOptimization requires `g2o` python bindings.\n"
                "Install g2o (system build) and its python module, or vendor your own bindings."
            ) from e

        self._g2o = g2o
        self.edge_vertices: set[tuple[int, int]] = set()
        self.num_loop_closures = 0

        self.optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(solver)

        self.se3_offset_id = 0
        se3_offset = g2o.ParameterSE3Offset()
        se3_offset.set_id(self.se3_offset_id)
        self.optimizer.add_parameter(se3_offset)

    @property
    def vertex_ids(self) -> List[int]:
        return sorted(list(self.optimizer.vertices().keys()))

    def add_vertex(self, vertex_id: int, pose: np.ndarray, *, fixed: bool = False) -> None:
        g2o = self._g2o
        v = g2o.VertexSE3()
        v.set_id(int(vertex_id))
        v.set_estimate(g2o.Isometry3d(pose))
        v.set_fixed(bool(fixed))
        self.optimizer.add_vertex(v)

    def add_edge(
        self,
        vertex_a: int,
        vertex_b: int,
        measurement: np.ndarray,
        *,
        information: np.ndarray | None = None,
        robust_kernel: object | None = None,
        is_loop_closure: bool = False,
    ) -> None:
        g2o = self._g2o
        if information is None:
            information = np.eye(6, dtype=float)

        self.edge_vertices.add((int(vertex_a), int(vertex_b)))
        if is_loop_closure:
            self.num_loop_closures += 1

        edge = g2o.EdgeSE3()
        edge.set_vertex(0, self.optimizer.vertex(int(vertex_a)))
        edge.set_vertex(1, self.optimizer.vertex(int(vertex_b)))
        edge.set_measurement(g2o.Isometry3d(measurement))
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        self.optimizer.add_edge(edge)

    def optimize(self, *, max_iterations: int = 100, verbose: bool = False) -> List[np.ndarray]:
        self.optimizer.initialize_optimization()
        self.optimizer.set_verbose(bool(verbose))
        self.optimizer.optimize(int(max_iterations))

        poses: List[np.ndarray] = []
        for vertex_id in self.vertex_ids:
            pose = self.optimizer.vertex(vertex_id).estimate().matrix()
            poses.append(pose)
        return poses
