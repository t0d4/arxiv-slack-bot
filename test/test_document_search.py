from ..app.models import DocumentHandler, Searcher


def test_document_search():
    searcher = Searcher(initial_query="abs: Deep Learning")
    result_docs = searcher.search()
    handler = DocumentHandler()
    handler.load_model(model_name_or_path="StabilityAI/stablelm-base-alpha-3b")
    docs = handler.summarize_documents(result_docs)
    handler.unload_model()
    for doc in docs:
        assert doc.key_points is not None
