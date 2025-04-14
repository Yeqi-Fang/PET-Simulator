# pet_simulator/simulator.py
import numpy as np
import multiprocessing as mp
from functools import partial
from .numba_utils import simulate_batch
from .utils import save_detector_lut

def process_batch(process_id: int, batch_size: int, shared_data: dict) -> np.ndarray:
    """
    Process a batch of events in a separate process.
    Use the process_id to seed the random number generator.
    """
    np.random.seed(process_id)
    return simulate_batch(
        batch_size,
        shared_data['image'],
        shared_data['shape'],
        shared_data['voxel_size'],
        shared_data['detector_positions'],
        shared_data['radius'],
        shared_data['num_rings'],
        shared_data['crystal_axial_spacing'],
        shared_data['cumsum_prob'],
        shared_data['save_events_pos']
    )

class PETSimulator:
    def __init__(self, geometry, image: np.ndarray, voxel_size: float, save_events_pos: bool = False):
        self.geometry = geometry
        # Use float32 to reduce memory usage
        self.image = image.astype(np.float32)
        self.voxel_size = voxel_size
        self.save_events_pos = save_events_pos
        self._calculate_detector_positions()
        # Pre-calculate probability distribution for event generation
        total_activity = np.sum(self.image)
        assert total_activity > 0, "Total activity must be greater than 0."
        print('total_activity', total_activity)
        self.probabilities = self.image.ravel() / total_activity
        self.cumsum_prob = np.cumsum(self.probabilities)

    def _calculate_detector_positions(self):
        """Calculate detector positions based on geometry parameters."""
        angles = np.linspace(0, 2 * np.pi, self.geometry.crystals_per_ring, endpoint=False)
        effective_angles = angles + np.pi / 2
        self.detector_positions = np.zeros(
            (self.geometry.num_rings * self.geometry.crystals_per_ring, 3),
            dtype=np.float32
        )
        for ring in range(self.geometry.num_rings):
            z_pos = - (ring - (self.geometry.num_rings - 1) / 2) * self.geometry.crystal_axial_spacing
            start_idx = ring * self.geometry.crystals_per_ring
            end_idx = (ring + 1) * self.geometry.crystals_per_ring
            self.detector_positions[start_idx:end_idx, 0] = self.geometry.radius * np.cos(effective_angles)
            self.detector_positions[start_idx:end_idx, 1] = self.geometry.radius * np.sin(effective_angles)
            self.detector_positions[start_idx:end_idx, 2] = z_pos

    def simulate_events(self, num_events: int, use_multiprocessing: bool = True) -> np.ndarray:
        """
        Simulate events.
        
        If use_multiprocessing=True, use multiple processes (pool). 
        Otherwise, run in a single process (loop).
        """
        shared_data = {
            'image': self.image,
            'shape': self.image.shape,
            'voxel_size': self.voxel_size,
            'detector_positions': self.detector_positions,
            'radius': self.geometry.radius,
            'num_rings': self.geometry.num_rings,
            'crystal_axial_spacing': self.geometry.crystal_axial_spacing,
            'cumsum_prob': self.cumsum_prob,
            'save_events_pos': self.save_events_pos
        }

        if use_multiprocessing:
            num_processes = 12  # or some custom logic
            batch_size = num_events // num_processes

            with mp.Pool(num_processes) as pool:
                process_batch_partial = partial(
                    process_batch,
                    batch_size=batch_size,
                    shared_data=shared_data
                )
                # try:
                results = pool.map(process_batch_partial, range(num_processes))
                # except mp.pool.MaybeEncodingError as e:
                #     # try again with the same config
                #     print(f"Error: {e}")
                #     results = pool.map(process_batch_partial, range(num_processes))
            total_events = np.vstack(results)
        else:
            # Single-process approach: just call simulate_batch directly in a loop.
            from .numba_utils import simulate_batch  # or reuse the import at top
            total_events_list = []
            # We still split into "batches" to keep memory usage consistent, if desired.
            # Or you can run one big call if you prefer.
            batch_size = num_events
            events = simulate_batch(
                batch_size,
                shared_data['image'],
                shared_data['shape'],
                shared_data['voxel_size'],
                shared_data['detector_positions'],
                shared_data['radius'],
                shared_data['num_rings'],
                shared_data['crystal_axial_spacing'],
                shared_data['cumsum_prob'],
                shared_data['save_events_pos']
            )
            total_events_list.append(events)
            total_events = np.vstack(total_events_list)

        return total_events

    def save_detector_positions(self, filename: str):
        """Save detector positions lookup table using the utils module."""
        save_detector_lut(filename, self.detector_positions)
