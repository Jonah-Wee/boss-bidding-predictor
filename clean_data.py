from pathlib import Path
import re

import numpy as np
import pandas as pd


RAW_DATA_DIRECTORY = Path("data/raw")
OUTPUT_FILE = Path("data/processed/freshmen_history.csv")

TARGET_WINDOW = "Incoming Freshmen Rnd 1 Win 1"
MINIMUM_VALID_BID = 10.0


def clean_column_name(column: object) -> str:
    cleaned = str(column).strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)

    return cleaned.strip("_")


def extract_academic_year(filename: str) -> str:
    match = re.search(
        r"20\d{2}[-_ ]?\d{2}",
        filename,
    )

    if match is None:
        return "unknown"

    return re.sub(
        r"[_ ]",
        "-",
        match.group(),
    )


def find_bidding_window_column(
    data: pd.DataFrame,
) -> str:
    possible_names = {
        "bidding_window",
        "biddingwindow",
        "bid_window",
    }

    for column in data.columns:
        if (
            clean_column_name(column)
            in possible_names
        ):
            return str(column)

    raise KeyError(
        "Could not find Bidding Window column."
    )


def load_and_filter_file(
    file_path: Path,
) -> pd.DataFrame:
    print(f"\nReading: {file_path.name}")

    data = pd.read_excel(file_path)

    print(
        f"Rows before filtering: {len(data)}"
    )

    bidding_window_column = (
        find_bidding_window_column(data)
    )

    windows = (
        data[bidding_window_column]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    filtered = data[
        windows == TARGET_WINDOW.casefold()
    ].copy()

    filtered.columns = [
        clean_column_name(column)
        for column in filtered.columns
    ]

    if "section" not in filtered.columns:
        raise KeyError(
            "Could not find section column."
        )

    # Keep only sections beginning with G.
    section_values = (
        filtered["section"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    filtered = filtered[
        section_values.str.startswith(
            "G",
            na=False,
        )
    ].copy()

    print(
        "Matching freshman G-section rows: "
        f"{len(filtered)}"
    )

    filtered["academic_year"] = (
        extract_academic_year(
            file_path.name
        )
    )

    text_columns = filtered.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        filtered[column] = (
            filtered[column]
            .astype("string")
            .str.strip()
        )

    filtered["section"] = (
        filtered["section"]
        .astype("string")
        .str.upper()
    )

    return filtered


def convert_numeric_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    numeric_columns = {
        "before_process_vacancy",
        "after_process_vacancy",
        "median_bid",
        "min_bid",
        "enrolled_students",
        "vacancy",
        "opening_vacancy",
        "d_i_c_e",
    }

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    return data


def create_effective_min_bid(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Price that would have been sufficient to obtain
    a seat.

    If seats remained after bidding, e$10 would
    have been sufficient.
    """

    data["effective_min_bid"] = np.where(
        data["after_process_vacancy"] > 0,
        MINIMUM_VALID_BID,
        data["min_bid"],
    )

    # A filled section with a recorded minimum below
    # e$10 is not treated as a valid price observation.
    invalid = (
        (
            data["after_process_vacancy"]
            <= 0
        )
        & (
            data["min_bid"]
            < MINIMUM_VALID_BID
        )
    )

    data.loc[
        invalid,
        "effective_min_bid",
    ] = np.nan

    return data


def main() -> None:
    excel_files = sorted(
        list(
            RAW_DATA_DIRECTORY.glob("*.xlsx")
        )
        + list(
            RAW_DATA_DIRECTORY.glob("*.xls")
        )
    )

    if not excel_files:
        raise FileNotFoundError(
            "No Excel files found in data/raw/."
        )

    print(
        f"Found {len(excel_files)} Excel files."
    )

    frames: list[pd.DataFrame] = []

    for file_path in excel_files:
        try:
            frame = load_and_filter_file(
                file_path
            )

            if not frame.empty:
                frames.append(frame)

        except Exception as error:
            print(
                f"Error processing "
                f"{file_path.name}: {error}"
            )

    if not frames:
        raise ValueError(
            "No matching data was found."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined = convert_numeric_columns(
        combined
    )

    combined = create_effective_min_bid(
        combined
    )

    combined["course_code"] = (
        combined["course_code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    combined["instructor"] = (
        combined["instructor"]
        .fillna("Unknown Instructor")
        .astype("string")
        .str.strip()
    )

    combined["description"] = (
        combined["description"]
        .fillna("Unknown Course")
        .astype("string")
        .str.strip()
    )

    combined["school_department"] = (
        combined["school_department"]
        .fillna("Unknown School")
        .astype("string")
        .str.strip()
    )

    combined = combined.drop_duplicates()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nCleaning complete")
    print("-----------------")
    print(
        f"Rows saved: {len(combined)}"
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()