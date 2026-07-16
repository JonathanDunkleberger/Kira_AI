"""dialogue_reader.py - overworld DIALOGUE READER (a new SENSORY input, additive).

Reads the active field-message string from FireRed's gStringVar block (verified ~
0x02021CD0-0x02021E00), decodes the Gen-3 charmap, and fires ONCE per NEW notable
message (the verified box-open fallback: trigger on the buffer CHANGING to a new
valid terminated string; the buffer goes stale after close, so we dedupe on change).
Accepted lines are filler-filtered and SALIENCE-TAGGED, then handed to two consumers
via injected hooks:
  on_dialogue(line, tags) -> her REACTION path (the harness POSTs it through the
                             EXISTING pokemon_event seam as kind 'dialogue')
  state['last_dialogue'/'last_tags'] -> her OBJECTIVE sense (a harness field)

Touches NO personality/voice code. Overworld only - BATTLE text (gDisplayedString
Battle) stays the battle engine's job.
"""

# ── where the active field message lives (verified live vs a known on-screen line) ──
GSTRINGVAR_LO = 0x02021CD0
GSTRINGVAR_HI = 0x02021E00          # ~0x130 bytes spanning gStringVar1..4

MIN_NOTABLE_LEN = 12                # shorter than this = filler / a name placeholder
MAX_UNKNOWN_RATIO = 0.15           # decoded run with more junk than this = not real text

# ── system / UI strings to SKIP (filler). Small NAMED blocklist, easy to extend.
# MULTI-WORD phrases only - short single words (yes/no/hp/pp) substring-matched real
# dialogue ("no" inside "now"), so filler is caught by these phrases + the length floor. ──
SYSTEM_STRINGS = (
    "do what", "what will", "use which one", "choose a pok", "choose pok",
    "or cancel", "got away safely", "there's a time and place",
    "would you like to save", "do you want to", "turn off", "press the",
)

# ── SALIENCE vocabulary - a STARTER set, kept in ONE obvious place to grow. ──
# A line can match several categories or none.
SALIENCE = {
    "GYM_LEADER": {"brock", "misty", "surge", "erika", "koga", "sabrina",
                   "blaine", "giovanni"},
    "TYPE": {"rock", "water", "grass", "fire", "electric", "poison", "flying",
             "bug", "normal", "ground", "psychic", "ghost", "ice", "dragon",
             "fighting", "fairy"},
    "PLACE": {"pallet", "viridian", "pewter", "cerulean", "vermilion", "lavender",
              "celadon", "saffron", "fuchsia", "cinnabar", "route", "forest",
              "city", "town", "cave", "gym", "road", "plateau", "mt", "league"},
    "ITEM": {"potion", "pokeball", "poke ball", "antidote", "repel", "ball",
             "map", "pokedex", "badge", "hm", "tm", "ether"},
}

# ── Gen-3 (FireRed US) character table - the ranges verified by decoding a real line ──
_CTRL = {0x00: " ", 0xFE: "\n", 0xAB: "!", 0xAC: "?", 0xAD: ".", 0xAE: "-",
         0xB0: "..", 0xB1: '"', 0xB2: '"', 0xB3: "'", 0xB4: "'", 0xBA: "/",
         0xB8: ",", 0xAF: "*"}


def decode(run: bytes):
    """Decode a 0xFF-terminated Gen-3 byte run -> (text, unknown_ratio)."""
    out, unknown = [], 0
    for x in run:
        if x == 0xFF:
            break
        if 0xBB <= x <= 0xD4:
            out.append(chr(ord("A") + x - 0xBB))
        elif 0xD5 <= x <= 0xEE:
            out.append(chr(ord("a") + x - 0xD5))
        elif 0xA1 <= x <= 0xAA:
            out.append(chr(ord("0") + x - 0xA1))
        elif x in _CTRL:
            out.append(_CTRL[x])
        elif x in (0xFA, 0xFB, 0xFC, 0xFD):   # scroll/paragraph/colour control codes
            out.append(" ")
        else:
            out.append(".")
            unknown += 1
    s = "".join(out)
    return s, (unknown / len(s) if s else 1.0)


def is_filler(line: str) -> bool:
    low = line.lower().strip()
    if len(low) < MIN_NOTABLE_LEN:
        return True
    return any(tok in low for tok in (t.lower() for t in SYSTEM_STRINGS))


def salience_tags(line: str):
    low = line.lower()
    return [cat for cat, words in SALIENCE.items() if any(w in low for w in words)]


class DialogueReader:
    def __init__(self, bridge, on_dialogue=None, state=None, log=print):
        self.b = bridge
        self.on_dialogue = on_dialogue        # (line, tags) -> reaction seam (harness posts)
        self.state = state if state is not None else {}   # objective-sense field
        self.log = log
        self._prev = None                     # last buffer content (change detection)

    ACTIVE_MSG = 0x02021D18      # gStringVar3 - tracks the active field message (verified
                                 # live: it followed the displayed dialogue box-by-box,
                                 # spanning into the gStringVar4 address when the line is
                                 # long). The block-wide 'longest run' instead locked onto
                                 # a PERSISTENT STALE expand-buffer string and masked
                                 # everything - this reads the actually-displayed line.

    def _read_buffer(self):
        """Active field message: decode from gStringVar3 until the 0xFF terminator."""
        s, junk = decode(self.b.read_bytes(self.ACTIVE_MSG, 0xC0))
        s = s.strip()
        return s if (s and junk <= MAX_UNKNOWN_RATIO) else ""

    def poll(self):
        """Call each step. Fires ONCE per new notable message; ignores stale/unchanged
        buffers (the verified box-open fallback). Returns (line, tags) or None."""
        cur = self._read_buffer()
        if cur == self._prev:                 # unchanged -> stale buffer, do not re-fire
            return None
        self._prev = cur                      # buffer CHANGED
        if not cur or is_filler(cur):
            return None                        # changed but filler -> skip (won't re-fire)
        tags = salience_tags(cur)
        flat = cur.replace("\n", " ")
        self.log(f"   [dialogue] +++ {flat!r}  tags={tags}")   # LOUD on accept
        self.state["last_dialogue"] = flat
        self.state["last_tags"] = tags
        if self.on_dialogue:
            self.on_dialogue(flat, tags)       # -> reaction seam (harness posts as 'dialogue')
        return (flat, tags)
