# game_mode_controller.py — Manages the "Gaming/Observer" mode state.

class GameModeController:
    """
    Lightweight state controller for Gaming/Observer mode.
    When active, enables vision heartbeat and media bridge monitoring.
    """
    def __init__(self, vision_agent, media_bridge=None):
        self.vision = vision_agent
        self.media_bridge = media_bridge
        self.is_active = False
