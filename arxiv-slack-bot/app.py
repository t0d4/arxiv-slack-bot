import os
import re

import config
import user_interfaces
import utils
from arxiv import SortCriterion
from dotenv import load_dotenv
from exceptions import DocumentAlreadyVectorizedException
from models import DocumentHandler, Searcher
from slack_bolt import Ack, App, BoltContext, Say
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.models.blocks import ContextBlock, MarkdownTextObject
from slack_sdk.web.client import WebClient

# load secret tokens to environmental variables
load_dotenv(dotenv_path=".env")
# Initialize app with BOT_TOKEN
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# initialize utility classes
doc_handler = DocumentHandler(
    llm_model_name_or_path=config.LLM_MODEL_NAME_OR_MODEL_PATH,
    deepl_token=os.environ["DEEPL_AUTH_TOKEN"],
)
searcher = Searcher(initial_query=config.INITIAL_QUERY)


# open modal when command is issued
@app.command("/searchnow")
def searchnow(
    ack: Ack,
    body: dict,
    client: WebClient,
    context: BoltContext,
):
    ack()
    modal = user_interfaces.SelectSearchMethodModal()
    # save channel_id as a session variable.
    # channel_id should be embedded in modal every time
    # so that channel_id can be retrievable in later handlers.
    modal.private_metadata = context.channel_id
    client.views_open(trigger_id=body["trigger_id"], view=modal)


# change input areas of the modal when radio button is selected
@app.action("search_method-select-action")
def update_search_modal(
    ack: Ack,
    body: dict,
    client: WebClient,
):
    ack()

    channel_id = body["view"]["private_metadata"]
    action = body["actions"][0]
    search_type: str = action["selected_option"]["value"]
    if search_type == "search_with_url":
        modal = user_interfaces.SearchThesisWithURLModal()
        modal.private_metadata = channel_id
        client.views_update(
            view_id=body["view"]["id"],
            hash=body["view"]["hash"],
            view=modal,
        )
    elif search_type == "search_with_conditions":
        modal = user_interfaces.SearchThesisWithConditionsModal()
        modal.private_metadata = channel_id
        client.views_update(
            view_id=body["view"]["id"],
            hash=body["view"]["hash"],
            view=modal,
        )


# receive submission of the modal
@app.view(re.compile("modal_search_thesis_with_.+"))
def handle_search_modal(ack: Ack, view: dict, body: dict, client: WebClient, say: Say):
    ack()

    user_id: str = body[""]
    channel_id: str = view["private_metadata"]
    search_type: str = view["callback_id"]
    if search_type == "modal_search_thesis_with_url":
        thesis_url: str = view["state"]["values"]["thesis-url-input-block"][
            "url-input-action"
        ]["value"]
        if not thesis_url.startswith("https://arxiv.org/abs/"):
            say(
                channel=channel_id,
                text="URLは `https://arxiv.org/abs/xxxx.xxxxx` の形式で指定してください。",
                blocks=[
                    ContextBlock(
                        elements=[
                            MarkdownTextObject(
                                text="URLは `https://arxiv.org/abs/xxxx.xxxxx` の形式で指定してください。"
                            )
                        ]
                    ),
                ],
            )
            return

        thesis_id = thesis_url.split("/")[-1]
        docs = searcher.search(id_list=[thesis_id])
        if not docs:
            say(
                channel=channel_id,
                text=f"IDが `{thesis_id}` の論文が見つかりませんでした。",
                blocks=[
                    ContextBlock(
                        elements=[
                            MarkdownTextObject(
                                text=f"IDが `{thesis_id}` の論文が見つかりませんでした。"
                            )
                        ]
                    ),
                ],
            )
            return

        client.chat_postEphemeral(
            channel=channel_id, user=user_id, text="論文を要約しています。少しお待ちください。"
        )
        doc = doc_handler.summarize_documents(docs)[0]
        say(
            channel=channel_id,
            text=doc.arxiv_doc.title,
            blocks=doc.get_formatted_message(),
        )
    elif search_type == "modal_search_thesis_with_conditions":
        (
            query,
            max_results,
        ) = utils.get_query_and_max_desired_results_from_modal_submission(
            response_view=view
        )
        docs = searcher.search(
            query=query, max_results=max_results, sort_by=SortCriterion.Relevance
        )
        if not docs:
            say(
                channel=channel_id,
                text="条件に合致する論文が見つかりませんでした。",
                blocks=[
                    ContextBlock(
                        elements=[MarkdownTextObject(text="条件に合致する論文が見つかりませんでした。")]
                    ),
                ],
            )
            return

        client.chat_postEphemeral(
            channel=channel_id, user=user_id, text="論文を要約しています。少しお待ちください。"
        )
        docs = doc_handler.summarize_documents(docs=docs)
        for doc in docs:
            say(
                channel=channel_id,
                text=doc.arxiv_doc.title,
                blocks=doc.get_formatted_message(),
            )


@app.action("discuss-button-action")
def process_document_for_discussion(ack: Ack, body: dict, say: Say):
    ack()

    thread_id = body["message"]["ts"]
    thesis_id = body["actions"][0]["value"]

    say(text=f"<@{body['user']['id']}> 論文を熟読しています。しばらくお待ちください...", thread_ts=thread_id)

    try:
        doc_handler.convert_pdf_into_vector_db(thread_id=thread_id, thesis_id=thesis_id)
    except DocumentAlreadyVectorizedException as e:
        print(e)
        say(
            text=f"<@{body['user']['id']}> この論文はすでに読み込んであります! このbotをメンションして何でも聞いてください。",
            thread_ts=thread_id,
        )
    else:
        say(
            text=f"<@{body['user']['id']}> 準備ができました! このbotをメンションして何でも聞いてください。",
            thread_ts=thread_id,
        )


@app.event("app_mention")
def talk_about_document(event: dict, context: BoltContext, say: Say):
    # ignore unrelated messages
    if "thread_ts" not in event:
        return
    if not context.channel_id:
        return

    # remove mention symbol from text
    question = re.sub(pattern="<@.+>", repl="", string=event["text"])
    thread_id = event["thread_ts"]
    try:
        answer, source_docs = doc_handler.answer_question_with_source_documents(
            thread_id=thread_id, question=question
        )
    except FileNotFoundError:  # when vector db was not found for this thesis
        say(text="まだこの論文はちゃんと読んでいないようです。先にDiscuss it!ボタンを押してください。", thread_ts=thread_id)
    else:
        pass
        # TODO: check LLM's output and return it to the user


if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
