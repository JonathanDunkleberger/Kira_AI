"""
theme.py — Kira Control Center color system
Cozy coffee shop palette: warm espresso darks, cream text, caramel accents.

DESIGN PRINCIPLE: three-step lightness depth (dark = warm, not cold).
    APP_BG  (darkest espresso)  →  CARD_BG  (mid roast)  →  CONTROL_BG (oak/mahogany)
Every surface sits on exactly one of these three planes. Planes separate
visually even in a dim streaming room because they use WARM browns, not
neutral greys. Text is high-contrast warm cream — readable at a glance.

Accent hierarchy:
    ACCENT  = caramel/amber  — primary interactive (ON toggles, GO, focus)
    SAKURA  = warm sage      — personality/brand secondary (or swap to clay rose)

USAGE (CustomTkinter):
    import theme as T
    app.configure(fg_color=T.APP_BG)
    card = ctk.CTkFrame(app, fg_color=T.CARD_BG, border_color=T.BORDER,
                        border_width=1, corner_radius=T.RADIUS_CARD)
    btn  = ctk.CTkButton(card, fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
                         text_color=T.ON_ACCENT, corner_radius=T.RADIUS_CTRL)
    ctk.CTkSwitch(card, progress_color=T.ACCENT, fg_color=T.CONTROL_BG)

CTk color values may be a single hex string or a ("light", "dark") tuple.
This app is dark-only, so single strings are used throughout.
"""

# ----------------------------------------------------------------------------
# 1. DEPTH PLANES  — three-step warm ladder (most important section)
# ----------------------------------------------------------------------------
APP_BG      = "#1C1410"   # darkest  — espresso/dark roast — window background void
CARD_BG     = "#28201A"   # mid      — dark mahogany — every panel/card/section
CONTROL_BG  = "#382C22"   # lightest — warm oak — inputs, dropdowns, inactive toggles

# 4th step for nested sub-panels (e.g. perception bar, VN Autopilot box).
SUBPANEL_BG = "#312518"   # between CARD and CONTROL — warm teak

BORDER      = "#4E3C2A"   # warm brown hairline borders / dividers
BORDER_SOFT = "#3E3025"   # quieter divider when full border is too loud

# ----------------------------------------------------------------------------
# 2. TEXT — warm cream family. HIGH CONTRAST against all three planes.
# ----------------------------------------------------------------------------
TEXT_PRIMARY   = "#F4EACC"   # warm oat/cream — main readable text, button labels
TEXT_SECONDARY = "#C4A06A"   # warm caramel — sub-labels, "since Kira spoke"
TEXT_MUTED     = "#8A7050"   # warm taupe — hints, placeholders, off-state text
PERCEPTION_TEXT = "#A88A5E"  # one step up from MUTED — italic perception/feed text
HEADER         = "#B8904A"   # warm brass/honey — SECTION HEADERS (uppercase)
TEXT_ON_DARK   = "#F4EACC"   # alias: text sitting on APP/CARD/CONTROL

# ----------------------------------------------------------------------------
# 3. ACCENTS — caramel (primary interactive) + sage (brand/personality)
#    Caramel = ON state, GO, focus, primary buttons, active toggles
#    Sage    = personality expression, emotion text, Kira identity elements
# ----------------------------------------------------------------------------
ACCENT          = "#C08840"  # warm caramel — primary action / ON state
ACCENT_HOVER    = "#D4A055"  # caramel hover (lighter, warmer)
ACCENT_ACTIVE   = "#A87030"  # caramel pressed (darker)
ON_ACCENT       = "#1C1410"  # text/icon ON a filled caramel button (espresso)

SAKURA          = "#7A9E6E"  # muted sage green — personality/brand secondary
SAKURA_HOVER    = "#8CB480"
SAKURA_SOFT     = "#6E8F62"  # slightly deeper sage for text on dark
ON_SAKURA       = "#0E1C0C"  # text on a filled sage button

# Accent tints for subtle fills (selected tab bg, focus ring, etc.)
ACCENT_TINT     = "#3A2C18"  # caramel-tinted dark fill (e.g. active tab bg)
SAKURA_TINT     = "#1E2E1A"  # sage-tinted dark fill

# ----------------------------------------------------------------------------
# 4. SEMANTIC — meaning only. Warm, muted. Keep these rare.
# ----------------------------------------------------------------------------
SUCCESS         = "#7AAD68"  # Start stream, healthy/active — warm olive-green
SUCCESS_HOVER   = "#8DC07C"
ON_SUCCESS      = "#0E1C0C"

DANGER          = "#C0614A"  # End stream, Interrupt — warm terracotta, not alarm-red
DANGER_HOVER    = "#D07560"
ON_DANGER       = "#200E0A"

WARNING         = "#C4943A"  # Mute, caution, stale data — warm amber-honey
WARNING_HOVER   = "#D8A84E"
ON_WARNING      = "#1E1608"

INFO            = "#7A9DB0"  # neutral informational status (optional)

# Status-dot / live-state colors for the bottom status bar
STATUS_OK       = SUCCESS
STATUS_OFF      = TEXT_MUTED
STATUS_WARN     = WARNING

# ----------------------------------------------------------------------------
# 5. SHAPE / SPACING TOKENS
# ----------------------------------------------------------------------------
RADIUS_CARD  = 14
RADIUS_CTRL  = 8
RADIUS_PILL  = 999

PAD_CARD     = 16
PAD_CTRL_X   = 14
PAD_CTRL_Y   = 8
GAP          = 12

BORDER_WIDTH = 1

FONT_FAMILY        = "Segoe UI"
SIZE_HEADER        = 11
SIZE_LABEL         = 12
SIZE_BODY          = 13
SIZE_BUTTON        = 13
SIZE_METRIC        = 22
SIZE_EMOTION       = 20

# ----------------------------------------------------------------------------
# 6. SEMANTIC ROLE MAP — intent aliases so call sites read by purpose.
#    Edit here to restyle; widget code never uses raw hex.
# ----------------------------------------------------------------------------
# Primary interactive button (GO, Ask Kira's Thoughts, etc.)
BTN_PRIMARY_FG      = ACCENT
BTN_PRIMARY_HOVER   = ACCENT_HOVER
BTN_PRIMARY_TEXT    = ON_ACCENT

# Secondary / outline button (Pause Model, Reload Personality, etc.)
BTN_SECONDARY_FG    = CONTROL_BG
BTN_SECONDARY_HOVER = "#4A3C2C"
BTN_SECONDARY_TEXT  = TEXT_PRIMARY
BTN_SECONDARY_BORDER= BORDER

# Start / End stream
BTN_START_FG        = SUCCESS
BTN_START_HOVER     = SUCCESS_HOVER
BTN_START_TEXT      = ON_SUCCESS
BTN_END_FG          = DANGER
BTN_END_HOVER       = DANGER_HOVER
BTN_END_TEXT        = ON_DANGER

# Interrupt (danger) / Mute (warning)
BTN_INTERRUPT_FG    = DANGER
BTN_INTERRUPT_HOVER = DANGER_HOVER
BTN_INTERRUPT_TEXT  = ON_DANGER
BTN_MUTE_FG         = WARNING
BTN_MUTE_HOVER      = WARNING_HOVER
BTN_MUTE_TEXT       = ON_WARNING

# Toggles / switches
SWITCH_ON           = ACCENT
SWITCH_TRACK_OFF    = CONTROL_BG

# Failsafe resume button
FAILSAFE_FG         = WARNING
FAILSAFE_HOVER      = WARNING_HOVER
FAILSAFE_TEXT       = ON_WARNING

SWITCH_KNOB         = TEXT_PRIMARY  # button_color (the knob)

# Inputs / dropdowns
INPUT_FG            = CONTROL_BG
INPUT_BORDER        = BORDER
INPUT_TEXT          = TEXT_PRIMARY
INPUT_PLACEHOLDER   = TEXT_MUTED

# Emotion readout text — sage (brand personality) on a card plane
EMOTION_TEXT        = SAKURA
EMOTION_BOX_FG      = CONTROL_BG

# Tabs (VN Autopilot / Media Watch)
TAB_ACTIVE_FG       = ACCENT_TINT
TAB_ACTIVE_TEXT     = TEXT_PRIMARY
TAB_INACTIVE_FG     = "transparent"
TAB_INACTIVE_TEXT   = TEXT_SECONDARY

# ----------------------------------------------------------------------------
# 7. ATTENTION — the ONE semantic "needs attention" color (desaturated rust).
#    Used EXCLUSIVELY for: errors, disabled-by-failure, and override-reason
#    chips. Amber (ACCENT) = active/engaged; rust (ATTENTION) = suppressed or
#    failed. Never mix the two meanings.
# ----------------------------------------------------------------------------
ATTENTION        = "#B5654A"  # desaturated rust — override / needs-attention
ATTENTION_DIM    = "#7E4636"  # dimmer rust — outline / low-emphasis attention
ATTENTION_TINT   = "#2E1A14"  # rust-tinted dark fill (override chip background)
ON_ATTENTION     = "#1C0F0A"  # text on a filled rust chip

# Three-state toggle rendering:
#   OFF      → SWITCH_TRACK_OFF track, TEXT_MUTED label
#   ON       → ACCENT track, TEXT_PRIMARY label
#   OVERRIDE → ACCENT outline + dimmed fill, rust reason chip
TOGGLE_OVERRIDE_FILL    = "#3A2C18"  # dimmed amber fill (ACCENT_TINT sibling)
TOGGLE_OVERRIDE_BORDER  = ACCENT     # amber outline says "you turned it on"


# ----------------------------------------------------------------------------
# 8. EMOTION BADGE TINTS — five distinct hues, all in the warm coffee family.
#    Same saturation level so no single badge screams; only the hue shifts.
# ----------------------------------------------------------------------------
EMOTION_HAPPY       = "#D4A24C"  # warm honey-gold
EMOTION_SASSY       = "#C77E5A"  # clay rose
EMOTION_MOODY       = "#9B8468"  # muted taupe (dim, withdrawn)
EMOTION_EMOTIONAL   = "#88B07A"  # soft sage
EMOTION_HYPERACTIVE = "#D98C3E"  # vivid amber-orange



