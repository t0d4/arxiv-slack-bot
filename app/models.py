import os
import re
import string
import textwrap
from typing import Optional

import config
import deepl
import torch  # arxiv.Result represents each thesis
from arxiv import Result, Search
from transformers import AutoModelForCausalLM, AutoTokenizer


class Document:
    def __init__(self, arxiv_doc: Result) -> None:
        self.arxiv_doc: Result = arxiv_doc
        self.key_points: Optional[list[str]] = None

    def __str__(self) -> str:
        return self.get_formatted_string()

    def get_formatted_string(self) -> str:
        """
        create string that contains information of this document.

        Note: Document must be summarized using `DocumentHandler.summarize_documents` before executing this method.

        Parameters:
            None

        Returns:
            str: formatted string that contains information of this document.
        """
        if not self.key_points:
            raise Exception("This document is not summarized yet.")

        formatted = textwrap.dedent(
            f"""
        [{self.arxiv_doc.entry_id.split("/")[-1]}]
        title: **{self.arxiv_doc.title}**
        link: {self.arxiv_doc.entry_id}
        summary:\
        """
        )
        for key_point in self.key_points:
            formatted += f"\n- {key_point}"
        return formatted


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
    def __init__(self) -> None:
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # load "path/to/stable-vicuna-13b-applied"
    def load_model(self, model_name_or_path) -> None:
        """
        load LLM to RAM or VRAM.

        Parameters:
            model_name_or_path: str - model name or path to the model, which is passed to `from_pretrained` method in `AutoClass` by hugging face transformers.

        Returns:
            None
        """
        if self._tokenizer and self._model:
            print(f"Model {self._model.config.model_name} is already loaded.")
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            device_map=config.HUGGING_FACE_DEVICE_MAP,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            device_map=config.HUGGING_FACE_DEVICE_MAP,
        )
        self._model.half().cuda()  # load model in half-precision
        print(f"Successfully loaded model: {self._model.config.model_name}")

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
        tokens = self._model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
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
            """\
        Read the abstract of delimited by triple backquotes and summarize it, \
        then write exactly 3 key points in the following output format:
        -- key point 1
        -- key point 2
        -- key point 3

        ```${abstract}```\
        """
        )

        for doc in docs:
            prompt = prompt_base.safe_substitute({"abstract": doc.arxiv_doc.summary})
            output = self._get_response_from_llm(prompt=prompt)
            key_points = re.findall(pattern="-- (.+)", string=output)
            doc.key_points = key_points

        if translate:
            docs = self._translate_key_points_of_documents(
                docs=docs, api_token=os.environ["DEEPL_AUTH_TOKEN"]
            )

        return docs
