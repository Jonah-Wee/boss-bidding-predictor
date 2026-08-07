from pathlib import Path
import json
import math

import streamlit as st


MODEL_FILE = Path(
    "models/predictor_model.json"
)


st.set_page_config(
    page_title="SMU BOSS Bid Predictor",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Prediction model not found."
        )

    with MODEL_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def round_bid(
    value: float,
) -> float:
    return (
        math.ceil(value * 100)
        / 100
    )


def get_prediction_parameters(
    model: dict,
    course_code: str,
    professor: str,
    school: str,
) -> tuple[dict, str]:
    """
    Choose the most specific available prediction.

    Priority:
    1. Same course + professor
    2. Same course
    3. Same professor
    4. Same school
    5. Global model
    """

    course_professor_model = (
        model[
            "course_professor_model"
        ]
    )

    if (
        course_code
        in course_professor_model
        and professor
        in course_professor_model[
            course_code
        ]
    ):
        return (
            course_professor_model[
                course_code
            ][professor],
            "Course and professor",
        )

    course_fallbacks = (
        model["course_fallbacks"]
    )

    if course_code in course_fallbacks:
        return (
            course_fallbacks[
                course_code
            ],
            "Course",
        )

    professor_fallbacks = (
        model[
            "professor_fallbacks"
        ]
    )

    if professor in professor_fallbacks:
        return (
            professor_fallbacks[
                professor
            ],
            "Professor",
        )

    school_fallbacks = (
        model[
            "school_fallbacks"
        ]
    )

    if school in school_fallbacks:
        return (
            school_fallbacks[
                school
            ],
            "School",
        )

    return (
        model["global_fallback"],
        "General model",
    )


def calculate_prediction(
    parameters: dict,
    current_vacancy: int,
    vacancy_adjustment_per_slot: float,
    minimum_bid: float,
) -> float:
    """
    More available slots reduce the predicted bid.

    Fewer available slots increase it.
    """

    base_bid = float(
        parameters["base_bid"]
    )

    average_vacancy = float(
        parameters[
            "average_vacancy"
        ]
    )

    vacancy_change = (
        current_vacancy
        - average_vacancy
    )

    predicted_bid = (
        base_bid
        - (
            vacancy_change
            * vacancy_adjustment_per_slot
        )
    )

    return max(
        minimum_bid,
        predicted_bid,
    )


try:
    model = load_model()

except Exception as error:
    st.error(
        f"Unable to load predictor: {error}"
    )
    st.stop()


minimum_bid = float(
    model.get(
        "minimum_valid_bid",
        10.0,
    )
)

vacancy_adjustment = float(
    model.get(
        "vacancy_adjustment_per_slot",
        0.25,
    )
)

course_metadata = model[
    "course_metadata"
]


st.title(
    "SMU BOSS Bid Predictor"
)

st.caption(
    "Freshman Round 1 Window 1 "
    "bid recommendation tool."
)

st.warning(
    "Predictions are estimates only and do not "
    "guarantee a successful bid."
)


course_options = sorted(
    course_metadata.keys()
)

selected_course = st.selectbox(
    "Choose a course",
    options=course_options,
)


metadata = course_metadata[
    selected_course
]

course_description = (
    metadata["description"]
)

school = metadata[
    "school"
]

st.write(
    f"**{selected_course} — "
    f"{course_description}**"
)


professor_options = (
    metadata["professors"]
)

selected_professor = st.selectbox(
    "Choose a professor",
    options=professor_options,
)


parameters, prediction_source = (
    get_prediction_parameters(
        model=model,
        course_code=selected_course,
        professor=selected_professor,
        school=school,
    )
)


default_vacancy = max(
    0,
    int(
        round(
            float(
                parameters[
                    "average_vacancy"
                ]
            )
        )
    ),
)

current_vacancy = st.number_input(
    "Slots available before bidding",
    min_value=0,
    value=default_vacancy,
    step=1,
    help=(
        "Enter the current "
        "'Before Process Vacancy' "
        "shown in BOSS."
    ),
)


predicted_bid = calculate_prediction(
    parameters=parameters,
    current_vacancy=int(
        current_vacancy
    ),
    vacancy_adjustment_per_slot=(
        vacancy_adjustment
    ),
    minimum_bid=minimum_bid,
)


uncertainty = float(
    parameters["uncertainty"]
)

confidence = str(
    parameters["confidence"]
)


economical_bid = round_bid(
    predicted_bid
)

balanced_bid = round_bid(
    max(
        minimum_bid,
        predicted_bid
        + uncertainty,
    )
)

safer_bid = round_bid(
    max(
        minimum_bid,
        predicted_bid
        + 2 * uncertainty,
    )
)


st.divider()


column_1, column_2 = st.columns(
    2
)

with column_1:
    st.metric(
        "Estimated bid needed",
        f"e${predicted_bid:.2f}",
    )

with column_2:
    st.metric(
        "Prediction confidence",
        confidence,
    )


st.subheader(
    "Suggested bids"
)

economical_column, balanced_column, safer_column = (
    st.columns(3)
)


with economical_column:
    st.metric(
        "Economical",
        f"e${economical_bid:.2f}",
    )

    st.caption(
        "Closest to the predicted "
        "minimum required bid."
    )


with balanced_column:
    st.metric(
        "Balanced",
        f"e${balanced_bid:.2f}",
    )

    st.caption(
        "Adds one prediction "
        "uncertainty margin."
    )


with safer_column:
    st.metric(
        "Safer",
        f"e${safer_bid:.2f}",
    )

    st.caption(
        "Adds two prediction "
        "uncertainty margins."
    )


if confidence == "Low":
    st.warning(
        "There is limited information for this "
        "course and professor combination. "
        "Treat this estimate cautiously."
    )


with st.expander(
    "How the predictor works"
):
    st.write(
        """
        The predictor considers the selected course,
        professor and number of available slots before
        bidding.

        Professor and course combinations receive the
        highest priority. When more slots are available,
        the predicted amount decreases; when fewer slots
        are available, the estimate increases.

        The website contains only derived prediction
        parameters. Individual historical BOSS bidding
        records are not included in the deployed app.

        The minimum recommendation is e$10.
        """
    )


st.caption(
    "Independent student-built tool. "
    "Not affiliated with or endorsed by SMU."
)