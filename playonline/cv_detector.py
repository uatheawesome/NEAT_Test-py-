import cv2
import numpy as np

class ScreenCVDetector:
    """
    Computer Vision detector to extract bird and pipe coordinates
    directly from screen / canvas image pixels.
    """
    def __init__(self, scale_to_sim=730.0 / 575.0):
        self.scale_to_sim = scale_to_sim
        # Flappy bird green pipe in HSV
        self.lower_green = np.array([35, 80, 80])
        self.upper_green = np.array([85, 255, 255])
        # Yellow bird in HSV
        self.lower_yellow = np.array([15, 120, 120])
        self.upper_yellow = np.array([35, 255, 255])

    def find_pipes_and_bird(self, bgr_image, crop_x_range=(370, 910)):
        """
        Processes a BGR screenshot image.
        Returns (bird_y, pipe_height, pipe_bottom) in simulation coordinates.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # 1. Pipe Detection
        pipe_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        col_counts = np.sum(pipe_mask > 0, axis=0)
        
        # Candidate pipe columns inside game canvas
        x_min, x_max = crop_x_range
        valid_cols = [c for c in range(x_min, min(x_max, len(col_counts))) if col_counts[c] > 80]
        
        pipe_top = 250.0 / self.scale_to_sim
        pipe_bottom = 450.0 / self.scale_to_sim
        
        if valid_cols:
            # Cluster pipe columns into distinct pipes
            clusters = []
            cur = [valid_cols[0]]
            for col in valid_cols[1:]:
                if col - cur[-1] <= 10:
                    cur.append(col)
                else:
                    clusters.append(cur)
                    cur = [col]
            clusters.append(cur)
            
            # Find the first pipe ahead
            for cluster in clusters:
                mid_x = cluster[len(cluster) // 2]
                col_slice = pipe_mask[:, mid_x]
                green_rows = np.where(col_slice > 0)[0]
                diffs = np.diff(green_rows)
                gap_idxs = np.where(diffs > 25)[0]
                if len(gap_idxs) > 0:
                    top_edge = green_rows[gap_idxs[0]]
                    bot_edge = green_rows[gap_idxs[0] + 1]
                    pipe_top = float(top_edge)
                    pipe_bottom = float(bot_edge)
                    break

        # 2. Bird Detection
        bird_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        # Search in the bird flight column range (~550 to ~650)
        bird_slice = bird_mask[:, 540:660]
        by_indices, _ = np.where(bird_slice > 0)
        
        if len(by_indices) > 0:
            bird_y = float(by_indices.mean())
        else:
            bird_y = 350.0 / self.scale_to_sim

        # Convert to training simulation scale (window height 800, floor 730)
        sim_bird_y = bird_y * self.scale_to_sim
        sim_pipe_top = pipe_top * self.scale_to_sim
        sim_pipe_bottom = pipe_bottom * self.scale_to_sim

        inputs = (
            sim_bird_y,
            abs(sim_bird_y - sim_pipe_top),
            abs(sim_bird_y - sim_pipe_bottom)
        )
        return inputs
