import json, os
res_p = 'data/output/function_calling_results.json'
def_p = 'data/input/functions_definition.json'
print(f'File exists: {os.path.exists(res_p)}')
if not os.path.exists(res_p): exit(1)
try:
    res = json.load(open(res_p))
    print('JSON valid: True')
except Exception as e:
    print(f'JSON valid: False ({e})')
    exit(1)
defs = json.load(open(def_p))
def_map = {d['name']: set(d['parameters']['properties'].keys()) for d in defs}
p, f, errs = 0, 0, []
for i, item in enumerate(res):
    try:
        if not all(k in item for k in ('prompt', 'name', 'parameters')):
             raise ValueError(f'Missing keys: {list(item.keys())}')
        name = item['name']
        if name not in def_map:
             raise ValueError(f'Unknown function: {name}')
        for k in item['parameters']:
            if k not in def_map[name]:
                 raise ValueError(f'Invalid param {k} for {name}')
        p += 1
    except Exception as e:
        f += 1
        if len(errs) < 3: errs.append(f'Item {i}: {e}')
print(f'Passes: {p}, Fails: {f}')
for e in errs: print(f'Sample failure: {e}')
