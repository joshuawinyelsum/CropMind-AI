import json, sys
path = sys.argv[1]
# The file has markdown frontmatter, skip to the JSON array
with open(path, "r", encoding="utf-8") as f:
    content = f.read()
# Find the JSON array
start = content.find("[{")
if start == -1:
    print("No JSON array found")
    sys.exit(1)
data = json.loads(content[start:])
for d in data:
    if d["type"] == "dir":
        print(d["name"])
