"""Audit source __all__ exports vs docs/api/*.md documentation coverage."""
import ast, os, re
from pathlib import Path

# --- 1. Extract ALL __all__ exports from source ---
source_exports = {}  # file_path -> {module, symbols}
root = Path('src/torchregress')
for pyfile in sorted(root.rglob('*.py')):
    try:
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == '__all__'
                    and isinstance(node.value, ast.List)):
                symbols = []
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        symbols.append(elt.value)
                if symbols:
                    rel = str(pyfile.relative_to(root).with_suffix('')).replace(os.sep, '.')
                    if rel.endswith('.__init__'):
                        rel = rel[:-9]
                    source_exports[str(pyfile)] = {'module': rel, 'symbols': symbols}
    except Exception:
        pass

# --- 2. Extract ALL documented symbols from API pages ---
api_symbols = {}  # api_page -> set of symbols

doc_patterns = [
    re.compile(r'`([A-Z][A-Za-z0-9_]+)`'),
    re.compile(r'`([a-z][a-z0-9_]*_[a-z][a-z0-9_]*)`'),
    re.compile(r'\|\s*`([A-Za-z_][A-Za-z0-9_]*)`\s*\|'),
]

false_positives = {
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
    'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
    'True', 'False', 'None', 'str', 'int', 'float', 'bool', 'dict', 'list',
    'tuple', 'set', 'any', 'all', 'len', 'min', 'max', 'sum', 'abs', 'zip',
    'Callable', 'Optional', 'Union', 'Tuple', 'List', 'Dict', 'Set',
    'Tensor', 'Module', 'Parameter', 'Optimizer', 'Protocol', 'Iterable',
    'Iterator', 'Sequence', 'Mapping', 'TypeVar', 'Generic', 'Literal',
    'Type', 'Any', 'Self', 'torch', 'numpy', 'np', 'pd', 'plt', 'NN', 'MSE',
    'MAE', 'RMSE', 'NLL', 'API', 'JSON', 'ML', 'AI', 'BNN', 'MDN',
    'ELU', 'ReLU', 'SGD', 'Adam', 'CPU', 'GPU', 'CI', 'PR',
    'README', 'MIT', 'BSD', 'Apache', 'WIP', 'TODO', 'FIXME',
}

doc_dir = Path('docs/api')
for apifile in sorted(doc_dir.glob('*.md')):
    text = apifile.read_text()
    symbols = set()
    for pattern in doc_patterns:
        for m in pattern.finditer(text):
            sym = m.group(1)
            if sym in false_positives:
                continue
            if len(sym) < 3 and not sym[0].isupper():
                continue
            if sym.isdigit():
                continue
            if all(c.isupper() for c in sym) and len(sym) <= 3:
                continue
            symbols.add(sym)
    api_symbols[apifile.name] = symbols

all_documented = set()
for syms in api_symbols.values():
    all_documented.update(syms)

# --- 3. Compare ---
total_exported = 0
total_missing = 0
missing_by_file = {}

for filepath, info in sorted(source_exports.items()):
    module = info['module']
    symbols = info['symbols']
    total_exported += len(symbols)
    missing = [s for s in symbols if s not in all_documented]
    total_missing += len(missing)
    if missing:
        missing_by_file[filepath] = {'module': module, 'missing': missing}

print("=" * 80)
print("UNDOCUMENTED EXPORTS")
print("=" * 80)

if missing_by_file:
    print(f"\n{total_missing} symbols in {len(missing_by_file)} files are NOT documented:\n")
    for filepath, info in missing_by_file.items():
        print(f"  {info['module']}:")
        for sym in info['missing']:
            print(f"    - {sym}")
        print()
else:
    print("\n  ✅ ALL EXPORTED SYMBOLS ARE DOCUMENTED")

print(f"\nSummary:")
print(f"  Total __all__ exports:   {total_exported}")
print(f"  Total doc symbols found: {len(all_documented)}")
print(f"  Missing from docs:       {total_missing}")
if total_exported > 0:
    print(f"  Coverage:                {100*(total_exported-total_missing)/total_exported:.1f}%")

# Also show which API page covers which modules
print("\n" + "=" * 80)
print("API PAGE → SYMBOL COUNT")
print("=" * 80)
for page in sorted(api_symbols.keys()):
    print(f"  {page:25s} {len(api_symbols[page]):4d} symbols")
