# game_mode_controller.py — Manages Observer / Activity mode state.

# Activity type constants
ACTIVITY_GENERAL    = "general"
ACTIVITY_VN         = "vn"       # Visual novel — enables autonomous gameplay loop
ACTIVITY_GAME       = "game"     # Generic game — observer only
ACTIVITY_MEDIA      = "media"    # Movie / anime / youtube watching


class GameModeController:
    """
    State controller for Observer mode and activity-aware context.
    When is_active=True the vision heartbeat runs.
    activity_type drives specialised behaviour (e.g. VN auto-play).
    """
    def __init__(self, vision_agent):
        self.vision = vision_agent
        self.is_active = False
        self.activity_type = ACTIVITY_GENERAL  # updated by bot activity detection

    def activate(self, activity_type: str = ACTIVITY_GENERAL):
        self.is_active = True
        self.activity_type = activity_type
        self.vision.is_active = True

    def deactivate(self):
        self.is_active = False
        self.activity_type = ACTIVITY_GENERAL
        self.vision.is_active = False
        # Clear stale visual context so get_vision_context() returns the
        # placeholder and _has_fresh_visual_context() returns False.
        # Without this, old scene_summary / last_description bleed into prompts
        # as ghost data even though no new captures are running.
        self.vision.scene_summary = ""
        self.vision.last_description = "I'm just getting my bearings. One sec!"
