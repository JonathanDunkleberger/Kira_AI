import json

with open('scripts/cuts_2026-06-13.json', encoding='utf-8') as f:
    cuts = json.load(f)

VOD_DUR = 16520  # 08:23:56 - 03:48:36 = 16520s

def ts_to_s(ts):
    h, m, s = ts.split(':')
    return int(h)*3600 + int(m)*60 + int(s)

ok, errors = 0, []
for c in cuts:
    s = ts_to_s(c['vod_anchor_ts'])
    start = s - c['pre']
    end   = s + c['post']
    if start < 0:
        errors.append(f"  id={c['id']} '{c['title']}': anchor {c['vod_anchor_ts']} - pre {c['pre']}s = NEGATIVE ({start}s)")
    elif end > VOD_DUR:
        errors.append(f"  id={c['id']} '{c['title']}': end {end}s > VOD {VOD_DUR}s")
    else:
        ok += 1

print(f"{ok}/{len(cuts)} clips in-range")
if errors:
    print("PROBLEMS:")
    for e in errors:
        print(e)
else:
    print("All timestamps valid.")

seen = {}
for c in cuts:
    s = ts_to_s(c['vod_anchor_ts'])
    if s in seen:
        print(f"DUPLICATE ANCHOR: id={c['id']} and id={seen[s]} both at {c['vod_anchor_ts']}")
    seen[s] = c['id']

bad_post = [c for c in cuts if c['post'] > 3]
if bad_post:
    print("POST > 3s:")
    for c in bad_post:
        print(f"  id={c['id']} post={c['post']}")
else:
    print("All post values <= 3s. OK")

print(f"Count: {len(cuts)} clips")
print(f"Scores: {sorted(set(c['score'] for c in cuts), reverse=True)}")
total = sum(c['pre'] + c['post'] for c in cuts)
print(f"Total clip material: {total}s = {total//60}m {total%60}s")
