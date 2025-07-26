from og_nav.mapping import OGMGenerator
import cv2
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class OGMScrollUI:
    """
    Interactive UI for occupancy grid map generation with parameter adjustment.

    This class provides a GUI interface with sliders for real-time
    adjustment of map generation parameters.
    """

    WINDOW_NAME = "Interactive OGM Generator"
    SLIDER_PARAMS = {
        "center_x": (0, -3000, 3000),  # -30.0 to 30.0
        "center_y": (0, -3000, 3000),  # -30.0 to 30.0
        "center_z": (0, -300, 300),  # -3.0 to 3.0
        "lower_x": (-1500, -3000, 0),  # -30.0 to 0.0
        "lower_y": (-1500, -3000, 0),  # -30.0 to 0.0
        "lower_z": (10, 0, 300),  # 0.0 to 3.0
        "upper_x": (1500, 0, 3000),  # 0.0 to 30.0
        "upper_y": (1500, 0, 3000),  # 0.0 to 30.0
        "upper_z": (100, 0, 300),  # 0.0 to 3.0
    }

    def __init__(self, generator: OGMGenerator) -> None:
        """
        Initialize interactive UI.

        Args:
            generator: OGM generator instance
        """
        self.generator = generator
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        logger.info("OGM Interactive UI initialized")

    def init_sliders(self) -> None:
        """Initialize parameter sliders."""
        try:
            for key, (init_val, min_val, max_val) in self.SLIDER_PARAMS.items():
                cv2.createTrackbar(
                    key,
                    self.WINDOW_NAME,
                    init_val - min_val,
                    max_val - min_val,
                    lambda x: None,
                )
            logger.info("Parameter sliders initialized")
        except Exception as e:
            logger.error(f"Error initializing sliders: {e}")

    def get_slider_values(self) -> Dict[str, float]:
        """
        Get current slider values.

        Returns:
            Dictionary of parameter values
        """
        values = {}
        try:
            for key, (_, min_val, _) in self.SLIDER_PARAMS.items():
                slider_val = cv2.getTrackbarPos(key, self.WINDOW_NAME)
                values[key] = (slider_val + min_val) / 100.0
        except Exception as e:
            logger.error(f"Error reading slider values: {e}")
        return values

    def generate_map(self, params: Dict[str, float]) -> None:
        """
        Generate and display map with current parameters.

        Args:
            params: Parameter dictionary from sliders
        """
        try:
            center = (params["center_x"], params["center_y"], params["center_z"])
            lower = (params["lower_x"], params["lower_y"], params["lower_z"])
            upper = (params["upper_x"], params["upper_y"], params["upper_z"])

            # Generate map
            grid_map = self.generator.generate_grid_map(
                center, lower, upper, return_img=True
            )

            # Prepare display
            display_img = self._prepare_display_image(grid_map, center, lower, upper)
            cv2.imshow(self.WINDOW_NAME, display_img)

            logger.debug("Map generated and displayed successfully")

        except Exception as e:
            logger.error(f"Map generation failed: {e}")
            self._show_error_message(str(e))

    def _prepare_display_image(
        self,
        grid_map: np.ndarray,
        center: Tuple[float, float, float],
        lower: Tuple[float, float, float],
        upper: Tuple[float, float, float],
    ) -> np.ndarray:
        """
        Prepare display image with enhanced size and information overlay.

        Args:
            grid_map: Generated occupancy grid map
            center: Map center coordinates
            lower: Lower bounds
            upper: Upper bounds

        Returns:
            Enhanced display image
        """
        # Get original dimensions
        height, width = grid_map.shape[:2]

        # Scale up for better visibility (minimum 800px width)
        scale_factor = max(1, 800 // width)
        new_width = width * scale_factor
        new_height = height * scale_factor

        # Resize image
        resized_map = cv2.resize(
            grid_map, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

        # Add information overlay
        info_height = 120
        display_img = np.zeros((new_height + info_height, new_width, 3), dtype=np.uint8)
        display_img[:new_height, :new_width] = resized_map

        # Add text information
        info_y_start = new_height + 20
        cv2.putText(
            display_img,
            f"Map Size: {width}x{height} (scaled {scale_factor}x)",
            (10, info_y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            display_img,
            f"Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})",
            (10, info_y_start + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            display_img,
            f"Range: ({lower[0]:.1f},{lower[1]:.1f},{lower[2]:.1f}) to ({upper[0]:.1f},{upper[1]:.1f},{upper[2]:.1f})",
            (10, info_y_start + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            display_img,
            "Controls: SPACE=Generate, R=Reset, Q=Quit",
            (10, info_y_start + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
        )

        return display_img

    def _show_error_message(self, error_msg: str) -> None:
        """
        Display error message in the window.

        Args:
            error_msg: Error message to display
        """
        error_img = np.zeros((300, 600, 3), dtype=np.uint8)
        cv2.putText(
            error_img,
            "Map Generation Error:",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            error_img,
            error_msg[:50],
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            error_img,
            "Check parameters and try again",
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            1,
        )
        cv2.imshow(self.WINDOW_NAME, error_img)
