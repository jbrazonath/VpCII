import nbformat

def fix_torch_load(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            # Finding the torch.load line even with partial matches or newlines
            if "torch.load(PATH" in cell.source and "map_location" not in cell.source:
                cell.source = cell.source.replace(
                    "checkpoint = torch.load(PATH, weights_only=False)",
                    "checkpoint = torch.load(PATH, map_location=torch.device('cpu'), weights_only=False)"
                )
    
    with open(file_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    fix_torch_load("GONZALEZ_MARTIN_DL_TP3_Co22.ipynb")
    print("Corregido el error de torch.load en el Notebook.")
