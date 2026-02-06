# game_mode_controller.py — Manages the "Observer" mode state.

class GameModeController:
    """
    Lightweight state controller for Observer mode.
    When active, enables the vision heartbeat.
    """
    def __init__(self, vision_agent):
        self.vision = vision_agent
        self.is_active = False
