import re

line = "[2024-01-01 12:00:00] INFO User123 Login success"
pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\w+) (\w+) (.+)'
match = re.match(pattern, line)
print(match.group(0))
print(match.groups())