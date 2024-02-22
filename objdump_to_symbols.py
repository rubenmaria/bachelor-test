import subprocess
import json

output = subprocess.run(['objdump', '-T', 'libc.so'], stdout=subprocess.PIPE).stdout.decode('utf-8')
output += subprocess.run(['objdump', '-T', 'libm.so'], stdout=subprocess.PIPE).stdout.decode('utf-8')

lines = output.splitlines()
lines = list(filter(lambda x: x != "", lines))

symbols = []
for line in lines:
    tokens = line.split()
    if len(tokens) < 7:
        continue
    symbols.append(line.split()[6])

with open('libm-libc-symbols.json', 'w', encoding='utf-8') as f:
    json.dump({"symbols": symbols}, f, ensure_ascii=False, indent=4)

