import gc
import os
import re
import string
import textwrap
from typing import Optional

import config
import deepl
import torch
from arxiv import Result, Search  # arxiv.Result represents each thesis
from slack_sdk.models.blocks import MarkdownTextObject, SectionBlock
from transformers import AutoModelForCausalLM, AutoTokenizer


class Document:
    def __init__(self, arxiv_doc: Result) -> None:
        self.arxiv_doc: Result = arxiv_doc
        self.key_points: Optional[list[str]] = None

    def __str__(self) -> str:
        return str(self.arxiv_doc)

    def get_formatted_message(self) -> list[SectionBlock]:
        """
        create slack message that contains information of this document.

        Note: Document must be summarized using `DocumentHandler.summarize_documents` before executing this method.

        Parameters:
            None

        Returns:
            list[Block]: slack message (consists of blocks) that contains information of this document.
        """
        if not self.key_points:
            raise Exception("This document is not summarized yet.")

        key_points_formatted = ""
        for key_point in self.key_points:
            key_points_formatted += f"• {key_point}\n"

        blocks = [
            SectionBlock(text=MarkdownTextObject(text=f"*[{self.arxiv_doc.title}]*")),
            SectionBlock(
                text=MarkdownTextObject(
                    text=f"<{self.arxiv_doc.entry_id}|view on arxiv.org>"
                )
            ),
            SectionBlock(text=MarkdownTextObject(text=key_points_formatted)),
        ]

        return blocks


class Searcher:
    def __init__(self, initial_query) -> None:
        self._query: str = initial_query

    def search(self, **kwargs) -> list[Document]:
        """
        search for theses on arXiv.

        Note: when both `query` and `id_list` is NOT specified, then search with `self._query`.

        Parameters:
            kwargs: keyword arguments passed to `arxiv.Seach.__init__()`.
        Returns:
            list[Documents]: list of `Document` instances which contains the search result
        """
        if "query" in kwargs or "id_list" in kwargs:
            search = Search(**kwargs)
        else:
            search = Search(query=self._query, **kwargs)
        return [Document(arxiv_doc=result) for result in search.results()]

    def update_query(self, new_query: str) -> None:
        """
        update default search query.
        after query is updated, `new_query` will be used if no query is specified for `search()`.

        Parameters:
            new_query: str - query string to replace current query.
        Returns:
            None
        """
        self._query = new_query


class DocumentHandler:
    def __init__(self, llm_model_name_or_path: str) -> None:
        """
        initialize `DocumentHandler` instance.

        Parameters:
            model_name_or_path: str - model name or path to the LLM, which will be passed to `from_pretrained` method in `AutoClass` by hugging face transformers.

        Returns:
            None
        """
        self._tokenizer = None
        self._model = None
        self.model_name_or_path = llm_model_name_or_path
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # load "path/to/stable-vicuna-13b-applied"
    def load_model(self) -> None:
        # TODO: add procedure to handle in case available VRAM is too small.

        """
        load LLM to RAM or VRAM.

        Parameters:
            None

        Returns:
            None
        """
        if self._tokenizer and self._model:
            print("Model is already loaded.")
            return

        print(f"Loading model: {self.model_name_or_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            device_map=config.HUGGING_FACE_DEVICE_MAP,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            device_map=config.HUGGING_FACE_DEVICE_MAP,
        )
        print("Successfully loaded the model")

    def unload_model(self) -> None:
        """
        unload LLM from RAM or VRAM.
        this method is designed to prevent `DocumentHandler` from occupying
        large amount of RAM or VRAM forever.

        Parameters:
            None

        Returns:
            None
        """
        # free the VRAM allocated for tokenizer and model
        # TODO: investigate if this works as expected
        self._tokenizer = None
        self._model = None
        gc.collect()
        print("Successfully unloaded the model")

    def _get_response_from_llm(self, prompt: str) -> str:
        if not self._tokenizer:
            raise Exception(
                "Tokenizer is not loaded yet. Call `load_model` to load it in advance."
            )
        if not self._model:
            raise Exception(
                "Model is not loaded yet. Call `load_model` to load it in advance."
            )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device=self.device)
        del inputs[
            "token_type_ids"
        ]  # this is currently necessary in order to avoid this error: ValueError: The following `model_kwargs` are not used by the model: ['token_type_ids'] (note: typos in the generate arguments will also show up in this list)
        tokens = self._model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_k=50,  # TODO: there's room for optimization
            top_p=0.95,
        )
        decoded_output = self._tokenizer.decode(tokens[0], skip_special_tokens=True)
        return decoded_output

    def _translate_key_points_of_documents(
        self, docs: list[Document], api_token: str
    ) -> list[Document]:
        translator = deepl.Translator(auth_key=api_token)
        for doc in docs:
            if not doc.key_points:
                raise Exception(
                    f"Document named {doc.arxiv_doc.title} is not summarized yet."
                )
            doc.key_points = [
                translation_result.text
                for translation_result in translator.translate_text(
                    text=doc.key_points, target_lang="ja"
                )  # type:ignore because translate_text MUST return List[TextResult]
                # when list[str] is passed to the argument "text"
            ]
        return docs

    def summarize_documents(
        self, docs: list[Document], translate=True
    ) -> list[Document]:
        """
        summarize documents by extracting their key points, and then
        write them to `Document.key_points`.

        Note: this method DESTRUCTIVELY changes the docs passed as argument.

        Parameter:
            docs: list[Document] - list of documents to process.
            translate: bool (default: True) - whether to translate key points into Japanese.

        Returns:
            docs: list[Document] - list of documents written their key points in `Document.key_points`
        """
        # TODO: consider better prompt
        prompt_base = string.Template(
            textwrap.dedent(
                """\
            You're a professional summary writer. Read the abstract of delimited by triple backquotes and summarize it, \
            then write exactly 3 key points in the output section indicated with [OUTPUT]

            ```${abstract}```

            [OUTPUT]
            -- key point 1:
            """
            )
        )

        for doc in docs:
            prompt = prompt_base.safe_substitute({"abstract": doc.arxiv_doc.summary})
            output = self._get_response_from_llm(prompt=prompt)
            key_points = re.findall(
                pattern=r"(?<=-- key point \d:)[\s\S]+?(?=--|$)", string=output
            )
            for idx, key_point in enumerate(key_points):
                key_points[idx] = key_point.strip()
            doc.key_points = key_points

        if translate:
            docs = self._translate_key_points_of_documents(
                docs=docs, api_token=os.environ["DEEPL_AUTH_TOKEN"]
            )

        return docs
