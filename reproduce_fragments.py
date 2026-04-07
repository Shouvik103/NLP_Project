from simplification import overall_simplify

text = "Each state and major mainland territory has its own legislature or parliament: unicameral in the Northern Territory, the ACT, and Queensland, and bicameral in the remaining states."

print(f"Original: {text}")
results = overall_simplify(text)
print("\nResults:")
for r in results:
    print(f"- {r}")
