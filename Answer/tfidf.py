from .pymu_extract import PyMuExtract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse._csr import csr_matrix
from numpy import ndarray
from nltk.corpus import stopwords

class TfidfVector:
    """Class responsible
    """

    def __init__(self, doc: PyMuExtract) -> None:
        """Load the tfIdfVectorizer and vectorize each page of PDFS

        Args:
            pdf (Pdf): a Pdf object
        """
        self.vector = TfidfVectorizer(stop_words=stopwords.words('french'))
        self.p_processed = []
        if isinstance(doc, PyMuExtract):
            self.paragrpahs = doc.paragraphs
        else:
            self.paragrpahs = doc.docs

        self.tfidf_docs = self._fit_transform(self.paragrpahs)
        # X = self.vector.fit_transform(pdf.pages)
        # print(X.shape)
        # print(self.vector.get_feature_names_out())

    def _fit_transform(self, pages: list[str]) -> csr_matrix:
        """Fit and transform the pages of the PDF in tfidf value

        Args:
            pages (list[str]): List of pages (text) in the PDF file

        Returns:
            csr_matrix: matrix of the fit transformed pages
        """
        return self.vector.fit_transform(pages)

    def _get_sorted_similarity(self, question: str) -> ndarray:
        """Get the index of the most similar pages of the given question

        Args:
            question (str): question asked

        Returns:
            ndarray: indexes of pages that are the most similar to the question asked
        """
        question_tftidf = self.vector.transform([question])
        cosine_similarities = cosine_similarity(question_tftidf, self.tfidf_docs, dense_output=True)
        return cosine_similarities.argsort()[0][::-1]

    def get_context_page(self, question: str) -> int:
        """Get the index of the page that will produce context for the given question

        Args:
            question (str): question asked

        Returns:
            int: the index of the page that contains the context
        """
        return self._get_sorted_similarity(question)[0]

    def get_top_paras(self, question : str, top: int) -> list[int]:
        return self._get_sorted_similarity(question)[:top]