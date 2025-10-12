from pathlib import Path
from typing import Iterable, List, Set

def collect_csv_logs(sources: Iterable[str], recursive: bool = False) -> Iterable[str]:
    paths: List[Path] = []
    for s in sources:
        p = Path(s)
        if p.is_file() and p.suffix.lower() == ".csv":
            paths.append(p)
        elif p.is_dir():
            it = p.rglob("*.csv") if recursive else p.glob("*.csv")
            paths.extend(it)
        else:
            paths.extend(Path().glob(s))
    uniq = sorted({str(p.resolve()) for p in paths if p.suffix.lower() == ".csv"})
    if not uniq:
        raise FileNotFoundError(f"No CSV files found in {list(sources)}")
    return uniq

def organize_by_name(
        all_logs: List[str],
        validation_basenames: Iterable[str],
        test_basenames: Iterable[str],
) -> tuple[List[str], List[str], List[str]]:
    """
    Take specified training and validation log files and group into training and validation sets
    Train on all other log files (i.e., train == everything not in test and validation sets)
    """

    def norm(names: Iterable[str]) -> Set[str]:
        out = set()
        for n in names:
            basename = Path(n).name
            out.add(basename[:-4] if basename.lower().endswith(".csv") else basename)
        return out

    validation_set = norm(validation_basenames)
    test_set = norm(test_basenames)

    train, validation, test = [], [], []
    for file in all_logs:
        stem = Path(file).stem
        if stem in validation_set:
            validation.append(file)
        elif stem in test_set:
            test.append(file)
        else:
            train.append(file)

    if not train:
        raise ValueError("Training split is empty. Check validation/test names.")
    return train, validation, test