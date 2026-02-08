import json
import os


def fix_notebook(filepath, replacements):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nb = json.load(f)

        changed = False
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                new_source = []
                for line in cell["source"]:
                    original_line = line
                    for old, new in replacements.items():
                        if old in line:
                            line = line.replace(old, new)
                    if line != original_line:
                        changed = True
                    new_source.append(line)
                cell["source"] = new_source

        if changed:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            print(f"Fixed {filepath}")
        else:
            print(f"No changes needed for {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")


# Fix Hawkes_MMD_Test.ipynb
hawkes_path = r"c:\Users\enzo.cAo\Documents\Projects\projet_recherche\New_LTPP\notebooks\Hawkes_MMD_Test.ipynb"
hawkes_replacements = {
    "from new_ltpp.evaluation.statistical_testing.test_factory import": "from new_ltpp.evaluation.statistical_testing import"
}
fix_notebook(hawkes_path, hawkes_replacements)

# Fix Test_MMD_Metric.ipynb
test_mmd_path = r"c:\Users\enzo.cAo\Documents\Projects\projet_recherche\New_LTPP\notebooks\Test_MMD_Metric.ipynb"
test_mmd_replacements = {
    "from new_ltpp.evaluation.statistical_testing.kernels.space_kernels import": "from new_ltpp.evaluation.statistical_testing.kernels import"
}
fix_notebook(test_mmd_path, test_mmd_replacements)
