from slack_sdk.models.blocks import (ActionsBlock, DividerBlock, InputBlock,
                                     MarkdownTextObject, Option,
                                     PlainTextInputElement, PlainTextObject,
                                     RadioButtonsElement, SectionBlock,
                                     StaticSelectElement, UrlInputElement)
from slack_sdk.models.views import View


class SelectSearchMethodModal(View):
    def __init__(self):
        super().__init__(
            type="modal",
            private_metadata="",  # filled later by handlers
            title=PlainTextObject(text="arXivから論文を検索"),
            submit=PlainTextObject(text="Submit"),
            close=PlainTextObject(text="Cancel"),
            blocks=[
                SectionBlock(text=PlainTextObject(text="論文のURLを指定するか、検索条件を指定してください。")),
                DividerBlock(),
                ActionsBlock(
                    elements=[
                        RadioButtonsElement(
                            options=[
                                Option(
                                    text=PlainTextObject(text="論文のURLを入力"),
                                    value="search_with_url",
                                ),
                                Option(
                                    text=PlainTextObject(text="検索条件を入力"),
                                    value="search_with_conditions",
                                ),
                            ],
                            action_id="search_method-select-action",
                        )
                    ]
                ),
            ],
        )


class SearchThesisWithURLModal(View):
    def __init__(self):
        super().__init__(
            type="modal",
            private_metadata="",  # filled later by handlers
            callback_id="modal_search_thesis_with_url",
            submit=PlainTextObject(text="Go"),
            close=PlainTextObject(text="Cancel"),
            title=PlainTextObject(text="URLで論文を検索"),
            blocks=[
                DividerBlock(),
                InputBlock(
                    block_id="thesis-url-input-block",
                    element=UrlInputElement(action_id="url-input-action"),
                    label=PlainTextObject(
                        text="arXiv上の論文のURL ( https://arxiv.org/abs/0123.4567 )"
                    ),
                ),
            ],
        )


class SearchThesisWithConditionsModal(View):
    def __init__(self):
        super().__init__(
            type="modal",
            private_metadata="",  # filled later by handlers
            callback_id="modal_search_thesis_with_conditions",
            submit=PlainTextObject(text="Submit"),
            close=PlainTextObject(text="Cancel"),
            title=PlainTextObject(text="条件で論文を検索"),
            blocks=[
                SectionBlock(text=PlainTextObject(text="複数条件が指定された場合はANDで接続されます。")),
                DividerBlock(),
                InputBlock(
                    block_id="title-input-block",
                    optional=True,
                    element=PlainTextInputElement(action_id="title-input-action"),
                    label=PlainTextObject(text="タイトル中のフレーズ"),
                ),
                InputBlock(
                    block_id="author-input-block",
                    optional=True,
                    element=PlainTextInputElement(action_id="author-input-action"),
                    label=PlainTextObject(text="著者名"),
                ),
                InputBlock(
                    block_id="abstract-input-block",
                    optional=True,
                    element=PlainTextInputElement(action_id="abstract-input-action"),
                    label=PlainTextObject(text="Abstract中のフレーズ"),
                ),
                InputBlock(
                    block_id="subject_category-input-block",
                    optional=True,
                    element=PlainTextInputElement(
                        action_id="subject_category-input-action"
                    ),
                    label=PlainTextObject(text="論文のカテゴリコード"),
                ),
                SectionBlock(
                    text=MarkdownTextObject(
                        text="使用可能なコードについては<https://arxiv.org/category_taxonomy|カテゴリコードの一覧>を参照"
                    )
                ),
                InputBlock(
                    block_id="max-results-input-block",
                    element=StaticSelectElement(
                        placeholder=PlainTextObject(text="取得する最大件数を選択"),
                        options=[
                            Option(text=PlainTextObject(text="3"), value="3"),
                            Option(text=PlainTextObject(text="5"), value="5"),
                            Option(text=PlainTextObject(text="10"), value="10"),
                        ],
                        action_id="max_results-static_select-action",
                    ),
                    label=PlainTextObject(text="取得する最大件数"),
                ),
            ],
        )
