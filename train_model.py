from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


DATA_FILE = Path(
    "data/processed/freshmen_history.csv"
)

MODEL_FILE = Path(
    "models/predictor_model.json"
)

MINIMUM_VALID_BID = 10.0

VACANCY_ADJUSTMENT_PER_SLOT = 0.25


def academic_year_to_number(
    value: object,
) -> int:
    match = re.search(
        r"20\d{2}",
        str(value),
    )

    if match is None:
        return 0

    return int(match.group())


def prepare_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    prepared = data.copy()

    numeric_columns = [
        "before_process_vacancy",
        "effective_min_bid",
    ]

    for column in numeric_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared["year_start"] = (
        prepared["academic_year"]
        .apply(
            academic_year_to_number
        )
    )

    prepared = prepared.dropna(
        subset=[
            "course_code",
            "instructor",
            "school_department",
            "before_process_vacancy",
            "effective_min_bid",
        ]
    )

    prepared["course_code"] = (
        prepared["course_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    prepared["instructor"] = (
        prepared["instructor"]
        .astype(str)
        .str.strip()
    )

    return prepared


def recency_weights(
    group: pd.DataFrame,
) -> np.ndarray:
    minimum_year = int(
        group["year_start"].min()
    )

    weights = (
        group["year_start"]
        - minimum_year
        + 1
    ).astype(float)

    return weights.to_numpy()


def summarise_group(
    group: pd.DataFrame,
) -> dict:
    """
    Convert historical rows into model parameters.

    Individual historical rows are NOT returned.
    """

    weights = recency_weights(
        group
    )

    base_bid = float(
        np.average(
            group["effective_min_bid"],
            weights=weights,
        )
    )

    average_vacancy = float(
        np.average(
            group[
                "before_process_vacancy"
            ],
            weights=weights,
        )
    )

    errors = (
        group["effective_min_bid"]
        - base_bid
    ).abs()

    uncertainty = float(
        np.average(
            errors,
            weights=weights,
        )
    )

    # Always provide at least e$1 of uncertainty.
    uncertainty = max(
        1.0,
        uncertainty,
    )

    observations = int(
        len(group)
    )

    if observations >= 3:
        confidence = "High"
    elif observations == 2:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "base_bid": round(
            max(
                MINIMUM_VALID_BID,
                base_bid,
            ),
            4,
        ),
        "average_vacancy": round(
            average_vacancy,
            4,
        ),
        "uncertainty": round(
            uncertainty,
            4,
        ),
        "confidence": confidence,
    }


def create_nested_model(
    data: pd.DataFrame,
) -> dict:
    """
    Create course -> professor model parameters.
    """

    result: dict = {}

    grouped = data.groupby(
        [
            "course_code",
            "instructor",
        ]
    )

    for (
        course_code,
        instructor,
    ), group in grouped:

        if course_code not in result:
            result[course_code] = {}

        result[course_code][instructor] = (
            summarise_group(group)
        )

    return result


def create_course_fallbacks(
    data: pd.DataFrame,
) -> dict:
    result: dict = {}

    for course_code, group in data.groupby(
        "course_code"
    ):
        result[course_code] = (
            summarise_group(group)
        )

    return result


def create_professor_fallbacks(
    data: pd.DataFrame,
) -> dict:
    result: dict = {}

    for instructor, group in data.groupby(
        "instructor"
    ):
        result[instructor] = (
            summarise_group(group)
        )

    return result


def create_school_fallbacks(
    data: pd.DataFrame,
) -> dict:
    result: dict = {}

    for school, group in data.groupby(
        "school_department"
    ):
        result[school] = (
            summarise_group(group)
        )

    return result


def create_course_metadata(
    data: pd.DataFrame,
) -> dict:
    """
    Store only information needed for the website's
    dropdowns.

    No bid history is included.
    """

    metadata: dict = {}

    for course_code, group in data.groupby(
        "course_code"
    ):
        latest = group.sort_values(
            "year_start",
            ascending=False,
        ).iloc[0]

        metadata[course_code] = {
            "description": str(
                latest["description"]
            ),
            "school": str(
                latest[
                    "school_department"
                ]
            ),
            "professors": sorted(
                group["instructor"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
        }

    return metadata


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. "
            "Run clean_data.py first."
        )

    data = pd.read_csv(
        DATA_FILE
    )

    data = prepare_data(
        data
    )

    if data.empty:
        raise ValueError(
            "No valid training data remains."
        )

    course_professor_model = (
        create_nested_model(
            data
        )
    )

    course_fallbacks = (
        create_course_fallbacks(
            data
        )
    )

    professor_fallbacks = (
        create_professor_fallbacks(
            data
        )
    )

    school_fallbacks = (
        create_school_fallbacks(
            data
        )
    )

    global_fallback = summarise_group(
        data
    )

    course_metadata = (
        create_course_metadata(
            data
        )
    )

    model = {
        "version": 1,
        "minimum_valid_bid": (
            MINIMUM_VALID_BID
        ),
        "vacancy_adjustment_per_slot": (
            VACANCY_ADJUSTMENT_PER_SLOT
        ),
        "course_metadata": (
            course_metadata
        ),
        "course_professor_model": (
            course_professor_model
        ),
        "course_fallbacks": (
            course_fallbacks
        ),
        "professor_fallbacks": (
            professor_fallbacks
        ),
        "school_fallbacks": (
            school_fallbacks
        ),
        "global_fallback": (
            global_fallback
        ),
    }

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MODEL_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model,
            file,
            indent=2,
        )

    print("\nModel created")
    print("-------------")
    print(
        f"Courses: "
        f"{len(course_metadata)}"
    )
    print(
        "Course-professor combinations: "
        f"{sum(len(value) for value in course_professor_model.values())}"
    )
    print(
        f"Saved to: {MODEL_FILE}"
    )

    print(
        "\nThe model file contains aggregated "
        "prediction parameters only."
    )

    print(
        "Individual historical BOSS rows are "
        "not included."
    )


if __name__ == "__main__":
    main()