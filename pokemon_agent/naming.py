"""naming.py - operate FireRed's name-entry KEYBOARD (capability, not decision).

Layout from the pokefirered disasm (naming_screen.c): the UPPERCASE page is a 4x8 grid, the cursor
starts at (0,0)='A', the OK/confirm button is a separate BUTTON COLUMN (START jumps the cursor
straight to OK), and the finished name commits to gSaveBlock2.playerName. This types ANY name; WHICH
name Kira picks is the CALLER's choice (a seam she/Batch-2 fills) - never hardcoded here.

Hard-won nav detail: LEFT from column 0 enters the BUTTON column (it does NOT clamp/wrap to the last
letter), which scrambled a naive 'reset to col 0' approach. So we TRACK the cursor and move by EXACT
deltas, vertical-first - every letter is col<=6, so vertical moves never land on a clamped short row.
Tracked-nav verified: KIRA/ZOE/NIDORAN/ASH all commit exactly.
"""

# UPPERCASE keyboard page (naming_screen.c sKeyboardChars[KEYBOARD_LETTERS_UPPER]); ' ' = a blank cell.
KB_GRID = ["ABCDEF .", "GHIJKL ,", "MNOPQRS", "TUVWXYZ"]
KB_POS = {ch: (c, r) for r, row in enumerate(KB_GRID) for c, ch in enumerate(row) if ch != " "}


def encode_gen3(name, maxlen=10):
    """Encode a name into Gen-3 (FireRed US) text bytes — the exact inverse of the verified decode
    ranges in dialogue_reader.decode (A-Z=0xBB.., a-z=0xD5.., 0-9=0xA1.., space=0x00). Unmappable
    characters are skipped. Returns exactly `maxlen` bytes: the encoded chars, a 0xFF terminator,
    then 0xFF padding (the game reads to the first 0xFF). Empty result -> None (nothing to write)."""
    out = []
    for ch in str(name):
        if len(out) >= maxlen:
            break
        if "A" <= ch <= "Z":
            out.append(0xBB + ord(ch) - ord("A"))
        elif "a" <= ch <= "z":
            out.append(0xD5 + ord(ch) - ord("a"))
        elif "0" <= ch <= "9":
            out.append(0xA1 + ord(ch) - ord("0"))
        elif ch == " " and out:                       # no leading spaces
            out.append(0x00)
    while out and out[-1] == 0x00:                    # no trailing spaces
        out.pop()
    if not out:
        return None
    return (out + [0xFF] * maxlen)[:maxlen]


def name_entry(b, name, render=None, owner="agent"):
    """Type `name` on the ALREADY-OPEN FireRed name keyboard, then confirm (START->OK, A). Letters
    only (A-Z); other chars are skipped. Returns nothing - the caller verifies via the committed
    gSaveBlock2.playerName. Assumes the 'What is your name?' prompt has been advanced into the
    keyboard."""
    render = render or (lambda: None)

    def tap(k):
        b.press(k, 8, 12, render, owner=owner)
        for _ in range(14):
            b.run_frame()
            render()

    cx = cy = 0                                      # cursor starts at (0,0)='A' (disasm)
    for ch in name.upper():
        if ch not in KB_POS:
            continue
        tx, ty = KB_POS[ch]
        while cy < ty:                               # vertical FIRST (col<=6 -> never clamps)
            tap("DOWN"); cy += 1
        while cy > ty:
            tap("UP"); cy -= 1
        while cx < tx:                               # then horizontal, by exact delta (no overshoot
            tap("RIGHT"); cx += 1                    # into the button column)
        while cx > tx:
            tap("LEFT"); cx -= 1
        tap("A")                                     # type the letter (cursor unchanged)
    tap("START")                                     # jump the cursor to OK
    b.press("A", 8, 12, render, owner=owner)         # confirm
    for _ in range(50):
        b.run_frame()
        render()
