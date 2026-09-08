from it_geek_search.query_parser import parse_query, MONTH_MAP
import re

query = 'What did Priya say in April 2026 about budget?'
cleaned = query.strip()
print("START:", cleaned)
cleaned = re.sub(r"\b(?:what did |from |by |posted by )?(Priya)(?:'s|\s+say|\s+mention|\s+ask|\s+post|\s+write)?\b", "", cleaned, flags=re.IGNORECASE)
print("AFTER SENDER:", cleaned)

for month_name, month_num in MONTH_MAP.items():
    month_pattern = rf"\b(?:in|around|back in|during|for)?\s*({month_name})(?:\s+(\d{{4}}))?\b"
    m = re.search(month_pattern, cleaned, re.IGNORECASE)
    if m:
        print(f"MATCHED MONTH LOOP: '{month_name}' with groups {m.groups()}")
        break

print("ACTUAL FUNCTION CALL:", parse_query(query))
