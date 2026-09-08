with open("it_geek_search/query_parser.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "print('DEBUG CLEANED" in line:
        continue
    if "print(f\"DEBUG PATTERN" in line:
        continue
    if "if month_name == 'april':" in line:
        continue
    if "print(f\"DEBUG IN FILE" in line:
        continue
    # Keep the fix: {{4}}
    new_lines.append(line)

with open("it_geek_search/query_parser.py", "w") as f:
    f.writelines(new_lines)
