# game_mode_controller.py — Manages Observer / Activity mode state.

# Activity type constants
ACTIVITY_GENERAL    = "general"
ACTIVITY_VN         = "vn"       # Visual novel — enables autonomous gameplay loop
ACTIVITY_GAME       = "game"     # Generic game — observer only
ACTIVITY_MEDIA      = "media"    # Movie / anime / youtube watching
ACTIVITY_MUSIC      = "music"    # Jonny playing/singing — react to HIM, audio=MUSIC

# Explicit-type set the dashboard category dropdown may arm directly (bypasses
# the keyword classifier). Keep in sync with _classify_activity_type's outputs.
ACTIVITY_TYPES = (ACTIVITY_GENERAL, ACTIVITY_VN, ACTIVITY_GAME, ACTIVITY_MEDIA, ACTIVITY_MUSIC)


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
        self.vision.master_enabled = True

    def deactivate(self):
        self.is_active = False
        self.activity_type = ACTIVITY_GENERAL
        self.vision.is_active = False
        # Master vision intent is now OFF — every on-demand capture path refuses.
        self.vision.master_enabled = False
        # Clear stale visual context so get_vision_context() returns the
        # placeholder and _has_fresh_visual_context() returns False.
        # Without this, old scene_summary / last_description bleed into prompts
        # as ghost data even though no new captures are running.
        self.vision.scene_summary = ""
        self.vision.last_description = "I'm just getting my bearings. One sec!"
        # Ghost-data rule applied everywhere: also wipe the rolling buffer and the
        # last-seen dialogue so nothing stale can leak after Vision is turned off.
        try:
            self.vision.context_buffer.buffer.clear()
        except Exception:
            pass
        self.vision.previous_dialogue = ""
        self.vision.last_capture_time = 0
