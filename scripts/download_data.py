"""Download the SkillPath datasets into data/raw/.

Kaggle datasets need an API token. See data/README.md for setup.

    python scripts/download_data.py --list        # show each dataset's files and sizes, download nothing
    python scripts/download_data.py               # download everything
    python scripts/download_data.py jobs onet     # download only some datasets
"""

import argparse
import io
import urllib.request
import zipfile

from skillpath.config import RAW_DIR

KAGGLE = {
    # key: (kaggle slug, why we use it)
    "jobs": ("asaniczka/1-3m-linkedin-jobs-and-skills-2024",
             "LinkedIn postings with descriptions and pre-extracted skills per job"),
    "courses": ("khusheekapoor/coursera-courses-dataset-2021",
                "Coursera courses with description, difficulty and skills"),
    "resumes": ("snehaanbhawal/resume-dataset",
                "Labeled resumes across 24 categories, used for evaluation test profiles"),
}
ONET_URL = "https://www.onetcenter.org/dl_files/database/db_30_2_text.zip"  # O*NET 30.2, CC BY 4.0


def kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi  # imported lazily: authenticates on import

    api = KaggleApi()
    api.authenticate()
    return api


def list_kaggle(api, slug):
    files = api.dataset_list_files(slug).files
    total = sum(f.total_bytes or 0 for f in files)
    print(f"{slug}  ({total / 1e6:,.1f} MB uncompressed)")
    for f in files:
        print(f"    {f.name:50s} {(f.total_bytes or 0) / 1e6:10,.1f} MB")


def download_onet():
    dest = RAW_DIR / "onet"
    print(f"O*NET 30.2 -> {dest}")
    with urllib.request.urlopen(ONET_URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(dest)


def main():
    parser = argparse.ArgumentParser()
    names = [*KAGGLE, "onet"]
    parser.add_argument("datasets", nargs="*", help=f"any of {names} (default: all)")
    parser.add_argument("--list", action="store_true", help="show files and sizes only")
    args = parser.parse_args()
    if unknown := set(args.datasets) - set(names):
        parser.error(f"unknown dataset(s) {sorted(unknown)}; choose from {names}")
    wanted = args.datasets or names

    kaggle_keys = [k for k in wanted if k in KAGGLE]
    api = kaggle_api() if kaggle_keys else None
    for key in kaggle_keys:
        slug, _ = KAGGLE[key]
        if args.list:
            list_kaggle(api, slug)
        else:
            print(f"{slug} -> {RAW_DIR / key}")
            api.dataset_download_files(slug, path=RAW_DIR / key, unzip=True, quiet=False)

    if "onet" in wanted:
        if args.list:
            print(f"O*NET 30.2 text files ({ONET_URL}), about 13 MB zipped")
        else:
            download_onet()


if __name__ == "__main__":
    main()
