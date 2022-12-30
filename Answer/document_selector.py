from os import listdir
from os.path import isfile, join
from .tfidf import TfidfVector
from. pymu_extract import PyMuExtract

class DocSelector:

    def __init__(self, db, fs) -> None:
        self.docs = self.get_docs_from_db(db, fs)
        self.vectorizer = TfidfVector(self)

    def get_docs(self):
        pdfs_content = []
        self.titles = [f.split('.')[0] for f in listdir('PDFs') if isfile(join('PDFs', f))][:-1]
        paths = [f'PDFs\\{f}' for f in listdir('PDFs') if isfile(join('PDFs', f))][:-1]
        for path in paths:
            #pdf = Pdf(path)
            pdf = PyMuExtract(path)
            #extract_content = " ".join(pdf.pages)
            extract_content = " ".join(pdf.paragraphs)
            pdfs_content.append(extract_content)
        return pdfs_content

    def get_docs_from_db(self, db, fs):
        self.files = list(db.fs.files.find({'name': {"$regex": ".*pdf$"}}))
        self.names = []
        pdfs_content = []
        for f in self.files:
            f_id = f['_id']
            self.names.append(f['name'])
            outputdata = fs.get(f_id).read()
            pdf = PyMuExtract(outputdata)
            extract_content = " ".join(pdf.paragraphs)
            pdfs_content.append(extract_content)
        return pdfs_content
        


    def get_selected_doc(self, q):
        paras_tfidf = self.vectorizer.get_top_paras(q, 1)[0]
        selected_doc = self.names[paras_tfidf]
        return selected_doc