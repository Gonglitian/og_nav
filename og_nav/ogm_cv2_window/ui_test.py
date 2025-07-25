import cv2
import numpy as np
from ui import OGMScrollUI


class MockOGMGenerator:
    """Mock OGM Generator for testing"""

    def __init__(self, resolution):
        self.resolution = resolution

    def generate_grid_map(self, map_center, lower_bound, upper_bound):
        """Generate simulated map for UI testing"""
        # Calculate map dimensions
        width = int(abs(upper_bound[0] - lower_bound[0]) * 20) + 100
        height = int(abs(upper_bound[1] - lower_bound[1]) * 20) + 100

        # Limit maximum size
        width = min(max(width, 200), 600)
        height = min(max(height, 200), 600)

        # Create colored test map (BGR format)
        test_map = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray background

        # Add some color blocks for display testing
        # Red region (obstacles)
        cv2.rectangle(test_map, (20, 20), (width // 3, height // 3), (0, 0, 255), -1)

        # Green region (passable)
        cv2.rectangle(
            test_map,
            (width // 2, height // 2),
            (width - 20, height - 20),
            (0, 255, 0),
            -1,
        )

        # Blue center point
        center_x, center_y = width // 2, height // 2
        cv2.circle(test_map, (center_x, center_y), 10, (255, 0, 0), -1)

        # Add parameter annotations
        cv2.putText(
            test_map,
            f"Center: {map_center}",
            (10, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            test_map,
            f"Size: {width}x{height}",
            (10, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            test_map,
            f"Range: {lower_bound} to {upper_bound}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )

        return test_map


def test_ogm_scroll_ui():
    """Test OGMScrollUI"""
    print("=== Test OGMScrollUI Fix Effects ===")
    print("Fixed content:")
    print("1. Remove incorrect color space conversion")
    print("2. Slider range extended to -30.0 ~ 30.0")
    print("3. Increase map display size to 800px")
    print("")
    print("Control instructions:")
    print("- Adjust sliders to test new range (-30~30)")
    print("- Press SPACE to generate test map")
    print("- Press 'r' to reset to default")
    print("- Press 'q' to quit")

    # Create mock generator
    mock_generator = MockOGMGenerator(0.1)

    # Create UI
    ui = OGMScrollUI(mock_generator)
    ui.init_sliders()

    # Display initial instruction image
    initial_img = np.ones((500, 900, 3), dtype=np.uint8) * 30

    instructions = [
        "OGMScrollUI Test Interface",
        "",
        "Fix Verification:",
        "✓ Bug Fix: Remove incorrect color conversion",
        "✓ Range Extension: Sliders now support -30.0 ~ 30.0",
        "✓ Display Enhancement: Map minimum 800px, clearer",
        "",
        "Test Steps:",
        "1. Adjust sliders to test new range",
        "2. Press SPACE to generate map",
        "3. Observe map size and clarity",
        "",
        "Controls: SPACE=Generate R=Reset Q=Quit",
    ]

    for i, text in enumerate(instructions):
        color = (255, 255, 255)
        if text.startswith("OGMScrollUI"):
            color = (0, 255, 255)  # Cyan title
        elif text.startswith("Fix Verification") or text.startswith("Test Steps"):
            color = (0, 200, 0)  # Green title
        elif text.startswith("✓"):
            color = (0, 255, 0)  # Green checkmark
        elif text.startswith("Controls"):
            color = (255, 200, 0)  # Orange control instructions

        cv2.putText(
            initial_img,
            text,
            (30, 50 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    cv2.imshow("Interactive OGM Generator", initial_img)

    def reset_sliders():
        """Reset sliders to default values"""
        for key, (init_val, min_val, max_val) in ui.SLIDER_PARAMS.items():
            cv2.setTrackbarPos(key, ui.WINDOW_NAME, init_val - min_val)
        print("[INFO] Sliders have been reset to default values")
        cv2.imshow("Interactive OGM Generator", initial_img)

    def print_current_values():
        """Print current slider values"""
        values = ui.get_slider_values()
        print(f"[DEBUG] Current parameter values:")
        for key, val in values.items():
            print(f"  {key}: {val:.2f}")

    # Main loop
    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            print("[INFO] Test completed")
            break
        elif key == ord(" "):
            print("[INFO] Generating test map...")
            print_current_values()
            params = ui.get_slider_values()
            ui.generate_map(params)
        elif key == ord("r"):
            reset_sliders()
        elif key == ord("d"):  # Debug mode
            print_current_values()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_ogm_scroll_ui()
