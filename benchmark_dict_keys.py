import timeit

setup_code = """
large_dict = {i: i for i in range(10000)}
"""

test_code_with_keys = """
for k in large_dict.keys():
    pass
"""

test_code_without_keys = """
for k in large_dict:
    pass
"""

time_with_keys = timeit.timeit(stmt=test_code_with_keys, setup=setup_code, number=10000)
time_without_keys = timeit.timeit(stmt=test_code_without_keys, setup=setup_code, number=10000)

print(f"Time with .keys(): {time_with_keys:.6f} seconds")
print(f"Time without .keys(): {time_without_keys:.6f} seconds")
print(f"Improvement: {(time_with_keys - time_without_keys) / time_with_keys * 100:.2f}%")
