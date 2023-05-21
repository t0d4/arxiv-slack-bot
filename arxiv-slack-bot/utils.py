def get_query_and_max_desired_results_from_modal_submission(
    response_view: dict,
) -> tuple[str, int]:
    query_string = ""
    state_values = response_view["state"]["values"]
    if title := state_values["title-input-block"]["title-input-action"]["value"]:
        title: str
        title = f'"{title}"'
        if query_string:
            query_string += f" AND ti:{title}"
        else:
            query_string += f"ti:{title}"

    if abstract := state_values["abstract-input-block"]["abstract-input-action"][
        "value"
    ]:
        abstract: str
        abstract = f'"{abstract}"'
        if query_string:
            query_string += f" AND abs:{abstract}"
        else:
            query_string += f"abs:{abstract}"

    if author := state_values["author-input-block"]["author-input-action"]["value"]:
        author: str
        author = f'"{author}"'
        if query_string:
            query_string += f" AND au:{author}"
        else:
            query_string += f"au:{author}"

    if category := state_values["subject_category-input-block"][
        "subject_category-input-action"
    ]["value"]:
        category: str
        if query_string:
            query_string += f" AND cat:{category}"
        else:
            query_string += f"cat:{category}"

    max_results_str = state_values["max-results-input-block"][
        "max_results-static_select-action"
    ]["selected_option"]["value"]
    max_results_str: str
    max_results = int(max_results_str)

    return query_string, max_results
