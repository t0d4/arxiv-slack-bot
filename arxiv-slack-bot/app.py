import asyncio
import os
import re

from arxiv import SortCriterion
from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncAck, AsyncApp, AsyncBoltContext, AsyncSay
from slack_sdk.models.blocks import ContextBlock, MarkdownTextObject
from slack_sdk.web.async_client import AsyncWebClient

import config
import user_interfaces
import utils
from models import DocumentHandler, Searcher

# load secret tokens to environmental variables
load_dotenv(dotenv_path=".env")
# Initialize app with BOT_TOKEN
app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))

# initialize utility classes
doc_handler = DocumentHandler(
    llm_model_name_or_path=config.LLM_MODEL_NAME_OR_MODEL_PATH
)
doc_handler.load_model()
searcher = Searcher(initial_query=config.INITIAL_QUERY)


# open modal when command is issued
@app.command("/searchnow")
async def searchnow(
    ack: AsyncAck,
    body: dict,
    client: AsyncWebClient,
    context: AsyncBoltContext,
):
    await ack()
    modal = user_interfaces.SelectSearchMethodModal()
    # save channel_id as a session variable.
    # channel_id should be embedded in modal every time
    # so that channel_id can be retrievable in later handlers.
    modal.private_metadata = context.channel_id
    await client.views_open(trigger_id=body["trigger_id"], view=modal)


# change input areas of the modal when radio button is selected
@app.action("search_method-select-action")
async def update_search_modal(
    ack: AsyncAck,
    body: dict,
    client: AsyncWebClient,
):
    await ack()

    channel_id = body["view"]["private_metadata"]
    action = body["actions"][0]
    selection: str = action["selected_option"]["value"]
    if selection == "search_with_url":
        modal = user_interfaces.SearchThesisWithURLModal()
        modal.private_metadata = channel_id
        await client.views_update(
            view_id=body["view"]["id"],
            hash=body["view"]["hash"],
            view=modal,
        )
    elif selection == "search_with_conditions":
        modal = user_interfaces.SearchThesisWithConditionsModal()
        modal.private_metadata = channel_id
        await client.views_update(
            view_id=body["view"]["id"],
            hash=body["view"]["hash"],
            view=modal,
        )


# receive submission of the modal
@app.view(re.compile("modal_search_thesis_with_.+"))
async def handle_search_modal(ack: AsyncAck, view: dict, say: AsyncSay):
    await ack()

    channel_id: str = view["private_metadata"]
    search_type: str = view["callback_id"]
    if search_type == "modal_search_thesis_with_url":
        thesis_url: str = view["state"]["values"]["thesis-url-input-block"][
            "url-input-action"
        ]["value"]
        if not thesis_url.startswith("https://arxiv.org/abs/"):
            await say(
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
            await say(
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
        doc_handler.load_model()
        doc = doc_handler.summarize_documents(docs)[0]
        await say(
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
        print(query)
        docs = searcher.search(
            query=query, max_results=max_results, sort_by=SortCriterion.Relevance
        )
        if not docs:
            await say(
                channel=channel_id,
                text="条件に合致する論文が見つかりませんでした。",
                blocks=[
                    ContextBlock(
                        elements=[MarkdownTextObject(text="条件に合致する論文が見つかりませんでした。")]
                    ),
                ],
            )
            return

        doc_handler.load_model()
        docs = doc_handler.summarize_documents(docs=docs)
        for doc in docs:
            await say(
                channel=channel_id,
                text=doc.arxiv_doc.title,
                blocks=doc.get_formatted_message(),
            )


async def main():
    handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
